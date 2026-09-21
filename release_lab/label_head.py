"""Undeployed serving prototype: retain only the original allowed-label rows.

The language backbone and the selected vocabulary weights are unchanged. This
restricted runtime cannot generate arbitrary text. Real-checkpoint probability
and latency qualification is required before using it in the public demo.
"""
import torch


class LabelProjection(torch.nn.Module):
    def __init__(self,head,label_ids):
        super().__init__()
        indices=torch.tensor(label_ids,dtype=torch.long,device=head.weight.device)
        self.weight=torch.nn.Parameter(head.weight.detach().index_select(0,indices).float(),requires_grad=False)
        self.bias=(torch.nn.Parameter(head.bias.detach().index_select(0,indices).float(),requires_grad=False)
                   if head.bias is not None else None)
        self.label_ids=tuple(label_ids)
        self.original_vocab_size=head.weight.shape[0]

    def forward(self,hidden):
        return torch.nn.functional.linear(hidden.float(),self.weight,self.bias)


def install(model,label_ids):
    head=model.get_output_embeddings()
    if isinstance(head,LabelProjection):raise ValueError('Label projection already installed')
    if not isinstance(head,torch.nn.Linear) or head.weight.requires_grad:
        raise ValueError('Only a frozen, ordinary vocabulary projection is supported')
    if head.weight.dtype!=torch.float32 or (head.bias is not None and head.bias.dtype!=torch.float32):
        raise ValueError('Qualify the existing float32 output projection first')
    if model.get_input_embeddings().weight.data_ptr()==head.weight.data_ptr():
        raise ValueError('Do not alter a tied input/output embedding')
    if (not 2<=len(label_ids)<=36 or len(set(label_ids))!=len(label_ids)
            or any(type(i) is not int or not 0<=i<head.weight.shape[0] for i in label_ids)):
        raise ValueError('Provide distinct in-range native label token identifiers')
    before=sum(p.numel()*p.element_size() for p in head.parameters())
    selected=LabelProjection(head,label_ids)
    model.set_output_embeddings(selected)
    after=sum(p.numel()*p.element_size() for p in selected.parameters())
    return dict(original_vocabulary_rows=head.weight.shape[0],retained_rows=len(label_ids),
                original_projection_bytes=before,restricted_projection_bytes=after,
                projection_bytes_removed=before-after,arbitrary_text_generation_supported=False,
                scope='Projection storage only; not measured total process memory or latency.')


def score(model,inputs,label_ids,option_mask):
    head=model.get_output_embeddings()
    if not isinstance(head,LabelProjection):raise ValueError('A qualified restricted projection is required')
    requested=tuple(label_ids.detach().cpu().tolist())
    if requested!=head.label_ids[:len(requested)]:raise ValueError('Label ordering differs from installed native rows')
    logits=model(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1,:]
    if logits.shape[-1]!=len(head.label_ids) or option_mask.shape!=(logits.shape[0],len(requested)):
        raise ValueError('Unexpected restricted output/menu shape')
    return logits[:,:len(requested)].float().masked_fill(~option_mask,-torch.inf)
