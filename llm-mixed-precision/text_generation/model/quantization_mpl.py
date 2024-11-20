import torch

def activation_quantize_fn(abitW):
    """
    MLP-specific quantization for activations with per-neuron scaling.
    """

    def quantize_activation(activation):
        qmin = - (2 ** (abitW - 1))
        qmax = (2 ** (abitW - 1)) - 1
        
        # Calculate scale per neuron (dimension 1) to capture per-neuron distribution
        scales = activation.abs().max(dim=1, keepdim=True)[0] / qmax
        activation_quantized = torch.round(activation / scales).clamp(qmin, qmax) * scales
        return activation_quantized

    return quantize_activation

def weight_quantize_fn(bitW):
    """
    MLP-specific quantization for weights with per-layer scaling.
    """
    def quantize_weight(weight):
        qmin = - (2 ** (bitW - 1))
        qmax = (2 ** (bitW - 1)) - 1
        
        # Calculate a single scale for the entire weight tensor (per layer)
        scale = weight.abs().max() / qmax
        weight_quantized = torch.round(weight / scale).clamp(qmin, qmax) * scale
        return weight_quantized
    
    return quantize_weight
