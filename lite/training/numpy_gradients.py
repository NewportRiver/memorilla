from __future__ import annotations

import numpy as np
from lite.neural.numpy_model import _gelu, _softmax

_TRAINABLE = ('score_slot_w', 'score_slot_b', 'score_mem_w', 'score_mem_b')


def _finite(x, name):
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f'{name} contains non-finite values')
    return x


def _normalize(x, axis=-1):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    if not np.isfinite(norm).all() or np.any(norm <= 1e-12):
        raise ValueError('scoring projection has zero or non-finite norm')
    return x / norm


def score_head_forward(latent, memory, state, valid_mask):
    """Return slot/evidence cosine scores using only the existing score head."""
    latent = _finite(latent, 'latent')
    memory = _finite(memory, 'memory')
    valid = np.asarray(valid_mask, dtype=bool)
    if latent.ndim != 2 or memory.ndim != 2 or valid.shape != (memory.shape[0],):
        raise ValueError('invalid scoring-head shapes')
    a = _normalize(latent @ state['score_slot_w'] + state['score_slot_b'])
    raw_b = memory @ state['score_mem_w'] + state['score_mem_b']
    b = np.zeros_like(raw_b, dtype=np.float32)
    if np.any(valid):
        b[valid] = _normalize(raw_b[valid])
    similarities = a @ b.T
    scores = np.full((memory.shape[0],), -np.inf, dtype=np.float32)
    slots = np.full((memory.shape[0],), -1, dtype=np.int64)
    for i in range(memory.shape[0]):
        if valid[i]:
            slots[i] = int(np.argmax(similarities[:, i]))
            scores[i] = similarities[slots[i], i]
    return scores, slots, (a, b, similarities)


def pairwise_loss_and_score_gradient(scores, positive_mask, valid_mask):
    """Multi-positive logistic ranking loss and d(loss)/d(scores)."""
    scores = _finite(np.where(np.isfinite(scores), scores, 0.0), 'scores')
    positive = np.asarray(positive_mask, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool)
    if scores.shape != positive.shape or scores.shape != valid.shape:
        raise ValueError('ranking mask shape mismatch')
    pos = np.flatnonzero(positive & valid)
    neg = np.flatnonzero(~positive & valid)
    grad = np.zeros_like(scores, dtype=np.float32)
    if not len(pos) or not len(neg):
        return np.float32(0.0), grad
    diffs = scores[neg, None] - scores[None, pos]
    loss = np.logaddexp(0.0, diffs).mean()
    sigmoid = 1.0 / (1.0 + np.exp(-diffs))
    grad[pos] -= sigmoid.sum(axis=0) / (len(pos) * len(neg))
    grad[neg] += sigmoid.sum(axis=1) / (len(pos) * len(neg))
    return np.float32(loss), grad


def scoring_head_gradients(latent, memory, state, positive_mask, valid_mask):
    """Compute exact Float32 gradients for the four existing scoring-head tensors."""
    scores, chosen, cache = score_head_forward(latent, memory, state, valid_mask)
    loss, dscore = pairwise_loss_and_score_gradient(scores, positive_mask, valid_mask)
    a, b, similarities = cache
    da = np.zeros_like(a, dtype=np.float64)
    db = np.zeros_like(b, dtype=np.float64)
    for i, slot in enumerate(chosen):
        if slot < 0 or dscore[i] == 0.0:
            continue
        g = float(dscore[i])
        av = a[slot].astype(np.float64)
        bv = b[i].astype(np.float64)
        da[slot] += g * (bv - similarities[slot, i] * av)
        db[i] += g * (av - similarities[slot, i] * bv)
    slot_raw = latent @ state['score_slot_w'] + state['score_slot_b']
    mem_raw = memory @ state['score_mem_w'] + state['score_mem_b']
    slot_norm = np.linalg.norm(slot_raw, axis=1, keepdims=True)
    mem_norm = np.linalg.norm(mem_raw, axis=1, keepdims=True)
    mem_norm = np.maximum(mem_norm, 1e-12)
    da_raw = (da - a.astype(np.float64) * (da * a.astype(np.float64)).sum(axis=1, keepdims=True)) / slot_norm
    db_raw = (db - b.astype(np.float64) * (db * b.astype(np.float64)).sum(axis=1, keepdims=True)) / mem_norm
    grads = {
        'score_slot_w': (latent.astype(np.float64).T @ da_raw).astype(np.float32),
        'score_slot_b': da_raw.sum(axis=0).astype(np.float32),
        'score_mem_w': (memory.astype(np.float64).T @ db_raw).astype(np.float32),
        'score_mem_b': db_raw.sum(axis=0).astype(np.float32),
    }
    if not all(np.isfinite(x).all() for x in grads.values()):
        raise FloatingPointError('non-finite scoring-head gradient')
    return loss, scores, grads


def _ln_backward(dy, x, gamma, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    centered = x - mean
    inv = 1.0 / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + eps)
    n = x.shape[-1]
    return (gamma * inv / n) * (n * dy - dy.sum(axis=-1, keepdims=True)
                                - centered * inv * inv * (dy * centered).sum(axis=-1, keepdims=True))


def _gelu_backward(dy, x):
    a = np.sqrt(2.0 / np.pi)
    t = a * (x + 0.044715 * x ** 3)
    return dy * (0.5 * (1.0 + np.tanh(t)) + 0.5 * x * (1.0 - np.tanh(t) ** 2) * a * (1.0 + 3.0 * 0.044715 * x ** 2))


def _score_head_latent_gradient(latent, memory, state, positive_mask, valid_mask):
    scores, chosen, cache = score_head_forward(latent, memory, state, valid_mask)
    loss, dscore = pairwise_loss_and_score_gradient(scores, positive_mask, valid_mask)
    a, b, similarities = cache
    da = np.zeros_like(a, dtype=np.float64)
    for i, slot in enumerate(chosen):
        if slot < 0 or dscore[i] == 0.0:
            continue
        g = float(dscore[i]); av = a[slot].astype(np.float64); bv = b[i].astype(np.float64)
        da[slot] += g * (bv - similarities[slot, i] * av)
    raw = latent @ state['score_slot_w'] + state['score_slot_b']
    norm = np.linalg.norm(raw, axis=1, keepdims=True)
    draw = (da - a.astype(np.float64) * (da * a.astype(np.float64)).sum(axis=1, keepdims=True)) / norm
    return loss, scores, (draw @ state['score_slot_w'].astype(np.float64).T).astype(np.float32)


def memory_projection_gradients(model, query, memory, positive_mask, valid_mask):
    """Backpropagate the ranking objective to only mem_k/mem_v initialization projections."""
    c, s = model.config, model.state
    query = _finite(query, 'query'); memory = _finite(memory, 'memory')
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != (memory.shape[0],) or not np.any(valid):
        zeros = {k: np.zeros_like(s[k]) for k in ('mem_k_w', 'mem_k_b', 'mem_v_w', 'mem_v_b')}
        return np.float32(0.0), np.full(memory.shape[0], -np.inf, np.float32), zeros, {'latent': np.zeros((c.slots, c.latent_dim), np.float32)}
    qlatent = query @ s['query_w'] + s['query_b']
    mk = memory @ s['mem_k_w'] + s['mem_k_b']; mv = memory @ s['mem_v_w'] + s['mem_v_b']
    summary = (mk[valid] + mv[valid]).mean(axis=0)
    slots = s['slots'] + qlatent[None, :] + summary[None, :]
    caches = []
    h, d = c.heads, c.latent_dim // c.heads
    for i in range(c.blocks):
        p = f'b{i}.'
        ln1 = (slots - slots.mean(axis=-1, keepdims=True)) / np.sqrt(((slots - slots.mean(axis=-1, keepdims=True)) ** 2).mean(axis=-1, keepdims=True) + 1e-5) * s[p+'ln1_g'] + s[p+'ln1_b']
        q = (ln1 @ s[p+'q_w'] + s[p+'q_b']).reshape(c.slots, h, d)
        k = (memory @ s[p+'k_w'] + s[p+'k_b']).reshape(-1, h, d)
        v = (memory @ s[p+'v_w'] + s[p+'v_b']).reshape(-1, h, d)
        logits = np.einsum('shd,nhd->shn', q, k) / np.sqrt(d)
        probs = np.zeros_like(logits); probs_flat = _softmax(logits.reshape(c.slots * h, -1), valid)
        probs[:] = probs_flat.reshape(c.slots, h, -1)
        ctx = np.einsum('shn,nhd->shd', probs, v).reshape(c.slots, c.latent_dim)
        slots_attn = slots + ctx @ s[p+'attn_out_w'] + s[p+'attn_out_b']
        centered = slots_attn - slots_attn.mean(axis=-1, keepdims=True)
        ln2 = centered / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + 1e-5) * s[p+'ln2_g'] + s[p+'ln2_b']
        ff1 = ln2 @ s[p+'ff1_w'] + s[p+'ff1_b']; ff = _gelu(ff1)
        slots_out = slots_attn + ff @ s[p+'ff2_w'] + s[p+'ff2_b']
        caches.append((slots, ln1, q, k, v, probs, slots_attn, ln2, ff1))
        slots = slots_out
    loss, scores, dslots = _score_head_latent_gradient(slots, memory, s, positive_mask, valid)
    for i in range(c.blocks - 1, -1, -1):
        p = f'b{i}.'; slots_in, ln1, q, k, v, probs, slots_attn, ln2, ff1 = caches[i]
        dslots_attn = dslots.copy()
        dff1 = _gelu_backward(dslots @ s[p+'ff2_w'].T, ff1)
        dln2 = _ln_backward(dff1 @ s[p+'ff1_w'].T, slots_attn, s[p+'ln2_g'])
        dslots_attn += dln2
        dctx = dslots_attn @ s[p+'attn_out_w'].T
        dctx = dctx.reshape(c.slots, h, d)
        dprobs = np.einsum('shd,nhd->shn', dctx, v)
        dv = np.einsum('shn,shd->nhd', probs, dctx)
        dlogits = probs * (dprobs - (dprobs * probs).sum(axis=-1, keepdims=True))
        dq = np.einsum('shn,nhd->shd', dlogits, k) / np.sqrt(d)
        dln1 = dq.reshape(c.slots, c.latent_dim) @ s[p+'q_w'].T
        dslots = dslots_attn + _ln_backward(dln1, slots_in, s[p+'ln1_g'])
    dsummary = dslots.sum(axis=0) / float(valid.sum())
    gk = np.zeros((memory.shape[0], c.latent_dim), dtype=np.float32); gv = np.zeros_like(gk)
    gk[valid] = dsummary; gv[valid] = dsummary
    grads = {'mem_k_w': (memory.astype(np.float64).T @ gk.astype(np.float64)).astype(np.float32), 'mem_k_b': gk.sum(axis=0).astype(np.float32), 'mem_v_w': (memory.astype(np.float64).T @ gv.astype(np.float64)).astype(np.float32), 'mem_v_b': gv.sum(axis=0).astype(np.float32)}
    if not all(np.isfinite(x).all() for x in grads.values()):
        raise FloatingPointError('non-finite memory projection gradient')
    return loss, scores, grads, {'latent': slots.copy(), 'dslots_initial': dslots.copy(), 'attention': [x[5].mean(axis=1) for x in caches]}


def clip_gradients(grads, max_norm):
    if max_norm <= 0:
        raise ValueError('max_norm must be positive')
    norm = float(np.sqrt(sum(np.square(x.astype(np.float64)).sum() for x in grads.values())))
    scale = min(1.0, max_norm / norm) if norm else 1.0
    return {k: (v * scale).astype(np.float32) for k, v in grads.items()}, norm, scale


def trainable_names():
    return _TRAINABLE


def _ln_param_grads(dy, x):
    mean = x.mean(axis=-1, keepdims=True)
    centered = x - mean
    inv = 1.0 / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + 1e-5)
    normed = centered * inv
    return (dy * normed).sum(axis=0).astype(np.float32), dy.sum(axis=0).astype(np.float32)


def associative_gradients(model, query, memory, positive_mask, valid_mask):
    """Exact native gradients for every tensor connected to the ranking objective.

    `out_w` and `out_b` are intentionally absent because the ranking objective consumes
    latent slots before the output projection.
    """
    c, s = model.config, model.state
    query = _finite(query, 'query'); memory = _finite(memory, 'memory')
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != (memory.shape[0],) or not np.any(valid):
        return np.float32(0.0), np.full(memory.shape[0], -np.inf, np.float32), {k: np.zeros_like(v) for k, v in s.items() if k not in ('out_w','out_b')}, {'latent': np.zeros((c.slots,c.latent_dim),np.float32)}
    qlatent = query @ s['query_w'] + s['query_b']
    mk = memory @ s['mem_k_w'] + s['mem_k_b']; mv = memory @ s['mem_v_w'] + s['mem_v_b']
    summary = (mk[valid] + mv[valid]).mean(axis=0)
    slots = s['slots'] + qlatent[None, :] + summary[None, :]
    caches = []
    h, d = c.heads, c.latent_dim // c.heads
    for i in range(c.blocks):
        p=f'b{i}.'
        x1=slots; m1=x1.mean(axis=-1,keepdims=True); r1=x1-m1; inv1=1/np.sqrt((r1*r1).mean(axis=-1,keepdims=True)+1e-5); n1=r1*inv1; ln1=n1*s[p+'ln1_g']+s[p+'ln1_b']
        q=(ln1@s[p+'q_w']+s[p+'q_b']).reshape(c.slots,h,d); k=(memory@s[p+'k_w']+s[p+'k_b']).reshape(-1,h,d); v=(memory@s[p+'v_w']+s[p+'v_b']).reshape(-1,h,d)
        logits=np.einsum('shd,nhd->shn',q,k)/np.sqrt(d); probs=_softmax(logits.reshape(c.slots*h,-1),valid).reshape(c.slots,h,-1); ctx3=np.einsum('shn,nhd->shd',probs,v); ctx=ctx3.reshape(c.slots,c.latent_dim); x2=x1+ctx@s[p+'attn_out_w']+s[p+'attn_out_b']
        m2=x2.mean(axis=-1,keepdims=True); r2=x2-m2; inv2=1/np.sqrt((r2*r2).mean(axis=-1,keepdims=True)+1e-5); n2=r2*inv2; ln2=n2*s[p+'ln2_g']+s[p+'ln2_b']; ff1=ln2@s[p+'ff1_w']+s[p+'ff1_b']; ff=_gelu(ff1); x3=x2+ff@s[p+'ff2_w']+s[p+'ff2_b']
        caches.append((x1,r1,inv1,n1,q,k,v,probs,ctx,x2,r2,inv2,n2,ln2,ff1)); slots=x3
    loss,scores,dslots=_score_head_latent_gradient(slots,memory,s,positive_mask,valid)
    grads={k:np.zeros_like(v) for k,v in s.items() if k not in ('out_w','out_b')}
    _, _, score_grads = scoring_head_gradients(slots, memory, s, positive_mask, valid)
    for name, value in score_grads.items():
        grads[name] = value.copy()
    for i in range(c.blocks-1,-1,-1):
        p=f'b{i}.'; x1,r1,inv1,n1,q,k,v,probs,ctx,x2,r2,inv2,n2,ln2,ff1=caches[i]
        dff1=_gelu_backward(dslots@s[p+'ff2_w'].T,ff1)
        grads[p+'ff2_w'] += ( _gelu(ff1).T @ dslots ).astype(np.float32); grads[p+'ff2_b'] += dslots.sum(axis=0).astype(np.float32)
        dln2=dff1@s[p+'ff1_w'].T; grads[p+'ff1_w'] += (ln2.T@dff1).astype(np.float32); grads[p+'ff1_b'] += dff1.sum(axis=0).astype(np.float32)
        grads[p+'ln2_g'] += (dln2*n2).sum(axis=0).astype(np.float32); grads[p+'ln2_b'] += dln2.sum(axis=0).astype(np.float32)
        dx2=dslots+_ln_backward(dln2,x2,s[p+'ln2_g'])
        grads[p+'attn_out_w'] += (ctx.T@dx2).astype(np.float32); grads[p+'attn_out_b'] += dx2.sum(axis=0).astype(np.float32)
        dctx=(dx2@s[p+'attn_out_w'].T).reshape(c.slots,h,d); dprobs=np.einsum('shd,nhd->shn',dctx,v); dv=np.einsum('shn,shd->nhd',probs,dctx); dlogits=probs*(dprobs-(dprobs*probs).sum(axis=-1,keepdims=True)); dq=np.einsum('shn,nhd->shd',dlogits,k)/np.sqrt(d); dk=np.einsum('shn,shd->nhd',dlogits,q)/np.sqrt(d)
        dq2=dq.reshape(c.slots,c.latent_dim); dk2=dk.reshape(-1,c.latent_dim); dv2=dv.reshape(-1,c.latent_dim)
        grads[p+'q_w'] += (ln1.T@dq2).astype(np.float32); grads[p+'q_b'] += dq2.sum(axis=0).astype(np.float32)
        grads[p+'k_w'] += (memory.T@dk2).astype(np.float32); grads[p+'k_b'] += dk2.sum(axis=0).astype(np.float32)
        grads[p+'v_w'] += (memory.T@dv2).astype(np.float32); grads[p+'v_b'] += dv2.sum(axis=0).astype(np.float32)
        dln1=dq2@s[p+'q_w'].T; grads[p+'ln1_g'] += (dln1*n1).sum(axis=0).astype(np.float32); grads[p+'ln1_b'] += dln1.sum(axis=0).astype(np.float32); dslots=dx2+_ln_backward(dln1,x1,s[p+'ln1_g'])
    grads['slots'] += dslots; grads['query_w'] += np.outer(query,dslots.sum(axis=0)).astype(np.float32); grads['query_b'] += dslots.sum(axis=0).astype(np.float32)
    dsummary=dslots.sum(axis=0)/float(valid.sum()); g=np.zeros((len(memory),c.latent_dim),np.float32); g[valid]=dsummary; grads['mem_k_w'] += (memory.T@g).astype(np.float32); grads['mem_k_b'] += g.sum(axis=0); grads['mem_v_w'] += (memory.T@g).astype(np.float32); grads['mem_v_b'] += g.sum(axis=0)
    if not all(np.isfinite(x).all() for x in grads.values()): raise FloatingPointError('non-finite associative gradient')
    return loss,scores,grads,{'latent':slots.copy(),'dslots_initial':dslots.copy(),'attention':[x[7].mean(axis=1) for x in caches]}
