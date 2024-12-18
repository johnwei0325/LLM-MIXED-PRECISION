'''
    code contributed by TaeHwan Jung(@graykode): https://github.com/graykode/gpt-2-Pytorch/blob/master/GPT2/model.py
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import copy
import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
from .quantization_mpl import *
from utils.options import args
from .models_utils import BaseLM
from transformers import GPT2Tokenizer
import torch.nn.functional as F

gpus = [int(gpu) for gpu in args.gpus.split(',')]

if gpus[0] != -1:
    device = torch.device(f"cuda:{gpus[0]}") 
else:
    device = 'cpu' 
    
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

# quant
class Linear(nn.Linear):
    """
    Class to quantize given linear layer weights
    """
    def __init__(self, in_features, out_features):
        """
        weight: bit-setting for weight
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or froze
        """
        super(Linear, self).__init__(in_features, out_features)

        self.quantize_fn = weight_quantize_fn(bitW=args.bitW)

        self.in_features = in_features
        self.out_features = out_features

        
    def forward(self, x):
        """
        using quantized weights to forward activation x
        """
        weight_q = self.quantize_fn(self.weight)
        return F.linear(x, weight=weight_q, bias=self.bias)
        
# quant
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        
        #self.quantize_fn = weight_quantize_fn(bitW=16)
        #self.act_q = activation_quantize_fn(abitW=16)
        
    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        #weight_q = self.quantize_fn(self.weight)

        #return weight_q * self.act_q(x) + self.bias
        return self.weight * x + self.bias

# quant    
class Conv1D(nn.Module):
    def __init__(self, nf, nx, bit):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))
        
        self.quantize_fn = weight_quantize_fn(bitW=bit)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.quantize_fn(self.weight))
        #x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

# quant
class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, bit, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx, bit)
        self.c_proj = Conv1D(n_state, nx, bit)
        
        self.act_q = activation_quantize_fn(abitW=bit)
     
    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e4 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        
        self.avg_q = torch.mean(torch.abs(query - self.act_q(query)))
        self.avg_q += torch.mean(torch.abs(key - self.act_q(key)))
        self.avg_q *= torch.mean(torch.abs(self.act_q(value)))
        
        query = self.act_q(query.float()).to(query.dtype)
        key = self.act_q(key.float()).to(query.dtype)
        value = self.act_q(value.float()).to(query.dtype)
        
        
        
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present


class MLP(nn.Module):
    def __init__(self, n_state, config, bit):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx, bit)
        self.c_proj = Conv1D(nx, n_state, bit)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2

class Block(nn.Module):
    def __init__(self, n_ctx, config, bit, scale=False):
        super(Block, self).__init__()
        self.n_ctx = n_ctx
        self.nx = config.n_embd
        self.config = config
        self.scale = scale
         
        self.ln_1 = LayerNorm(self.nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(self.nx, n_ctx, config, bit, scale)
        self.ln_2 = LayerNorm(self.nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * self.nx, config, bit)

    def forward(self, x, layer_past=None):
        float_attn = Attention(self.nx, self.n_ctx, 
                               self.config, 32, self.scale).to(device).float()
        
        
        float_a, _ = float_attn(self.ln_1(x).float(), layer_past=layer_past)
        #self.ln_1_cpu = self.ln_1.to('cpu')
        #float_a_cpu, _ = float_attn_cpu(self.ln_1_cpu(x.to('cpu')), layer_past=layer_past)
        #print("Input type to float_attn_cuda:", float_a_cuda.dtype)
        #print("Input type to float_attn_cpu:", float_a_cpu.dtype)

        #print('float a',  float_a)
        
        a, present = self.attn(self.ln_1(x), layer_past=layer_past)
        #print('=============================', a, present)
        self.eps_a = 1 / torch.numel(a) * torch.abs(torch.sum((a - float_a)**2) * torch.vmap(torch.trace)(float_a))
        #print(self.eps_a, x)       
        x = x + a
        
        m = self.mlp(self.ln_2(x))
    
        x = x + m

        return x, present

class GPT2Model(nn.Module):
    def __init__(self, config, bitconfig=None):
        super(GPT2Model, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        #block = Block(config.n_ctx, config, scale=True)
        if bitconfig is not None:
            self.h = nn.ModuleList([copy.deepcopy(Block(config.n_ctx, config, bitconfig[i], scale=True)) for i in range(config.n_layer)])
        else:
            self.h = nn.ModuleList([copy.deepcopy(Block(config.n_ctx, config, args.bitW, scale=True)) for i in range(config.n_layer)])
        
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = Linear(embed_shape[1], embed_shape[0])
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, input_ids, position_ids=None, token_type_ids=None, past=None, inputs_embeds=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))
        
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
            
        #self.inputs_embeds = inputs_embeds
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents

class GPT2LMHead(nn.Module):
    def __init__(self, model_embeddings_weights, config):
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = Linear(embed_shape[1], embed_shape[0])
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        lm_logits = self.decoder(hidden_state)
        return lm_logits

class GPT2LMHeadModel(BaseLM):
    def __init__(self, config, bitconfig=None):
        super().__init__()
        self.transformer = GPT2Model(config, bitconfig)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        self.vocab_size = self.tokenizer.vocab_size

        self._max_length = args.max_seq_len
        self.training = True

    def eval(self):
        """ Sets the model to evaluation mode """
        self.training = False
        self.transformer.eval()  # Set transformer to eval mode
        self.lm_head.eval()
    
    def train(self):
        """ Sets the model to training mode """
        self.training = True
        self.transformer.train()  # Set transformer to train mode
        self.lm_head.train()

    def to(self, device):
        """
        Moves the model to the specified device.
        
        Args:
            device (torch.device or str): The device to move the model to (e.g., 'cuda:0' or 'cpu').
        """
        self.transformer.to(device)
        self.lm_head.to(device)
        return self

    def set_tied(self):
        """ Make sure we are sharing the embeddings
        """
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, past=None, inputs_embeds=None):
        self.hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past, inputs_embeds)
        lm_logits = self.lm_head(self.hidden_states)
        if lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            return loss, lm_logits
        return lm_logits, presents
    
    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length: # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH


    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return args.eval_batch_size

    @property
    def device(self):
        if args.gpus[0] != -1:
            device = torch.device(f"cuda:{args.gpus[0]}") 
        else:
            device = 'cpu' 
        return device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.forward(inps)[0]

    def _model_generate(self, context, max_length, eos_token_id):
        generation_kwargs = {"do_sample": False, "max_length": max_length}
        if eos_token_id is not None:
            generation_kwargs['eos_token_id'] = eos_token_id
            generation_kwargs['pad_token_id'] = eos_token_id # setting eos_token_id as pad token
        return self.model.generate(context, **generation_kwargs)