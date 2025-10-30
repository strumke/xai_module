import torch

def make_activation_hook(activations): 
    def fwd_hook(module, input, output):
        out = output[0] 
        activations.append(out.squeeze(0).detach().cpu())
    return fwd_hook

def make_gradient_hook(grads):
    def bwd_hook(module, grad_input, grad_output):
        grad = grad_output[0] 
        grads.append(grad.squeeze(0).detach().cpu())
    return bwd_hook

def get_activations(model, data_loader, device, layer):
    model.eval()
    all_activations = []

    for batch_inputs in data_loader:
        for i in range(batch_inputs.size(0)):
            sample_input = batch_inputs[i].unsqueeze(0).to(device)

            activations = []
            hook = layer.register_forward_hook(make_activation_hook(activations))

            with torch.no_grad():
                _ = model(sample_input) 

            hook.remove()

            all_activations.append(activations[0])

    return torch.stack(all_activations)

def get_grads(model, data_loader, device, layer, target_idx=None):
    model.eval()
    all_grads = []

    for batch_inputs in data_loader:
        for i in range(batch_inputs.size(0)):
            sample_input = batch_inputs[i].unsqueeze(0).to(device)

            grads = []
            hook = layer.register_full_backward_hook(make_gradient_hook(grads))

            model.zero_grad()
            output = model(sample_input) 

            logits = output[0]

            if target_idx is None:
                logit = logits[logits.argmax()]
            else:
                logit = logits[target_idx]

            logit.backward()

            hook.remove()

            all_grads.append(grads[0])

    return torch.stack(all_grads)

def flatten(acts):
    return acts.view(acts.size(0), -1).numpy()

