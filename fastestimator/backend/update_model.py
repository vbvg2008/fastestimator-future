import tensorflow as tf
import torch


def update_model(model, loss, tape=None):
    loss = _reduce_loss(loss)
    if isinstance(model, tf.keras.Model):
        with tape.stop_recording():
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    elif isinstance(model, torch.nn.Module):
        loss.backward()
        model.optimizer.step()
    else:
        raise ValueError("Unrecognized model instance {}".format(type(model)))


def _reduce_loss(loss):
    if isinstance(loss, tf.Tensor):
        assert loss.ndim < 2, "loss must be one-dimentional or scalar"
        if loss.ndim == 1:
            loss = tf.reduce_sum(loss)
    elif isinstance(loss, torch.Tensor):
        assert loss.ndim < 2, "loss must be one-dimentional or scalar"
        if loss.ndim == 1:
            loss = torch.sum(loss)
    else:
        raise ValueError("loss must be either tf.Tensor or torch.Tensor")
    return loss
