import torch
import torch.nn as nn
from einops import repeat, rearrange
from math import pi

# --- Helper Functions (기존과 동일하거나 필요 함수 추가) ---
def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')

# --- [핵심 수정] Dynamic VisionRotaryEmbeddingFast ---
class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None, # 이제 이 값은 무시해도 됩니다 (자동 계산)
        custom_freqs=None,
        freqs_for='lang',
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        self.dim = dim
        self.custom_freqs = custom_freqs
        self.freqs_for = freqs_for
        self.theta = theta
        self.max_freq = max_freq
        self.num_freqs = num_freqs
        
        # 캐시 초기화 (빈 텐서)
        self.register_buffer("freqs_cos", torch.tensor([]), persistent=False)
        self.register_buffer("freqs_sin", torch.tensor([]), persistent=False)

    def forward(self, t, h, w):
        # 1. 입력 크기 확인
        batch, seq_len, head_dim = t.shape
        
        # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5')
        # print(t.shape)

        # [핵심 수정 1] CLS 토큰 유무 판별 (홀/짝이 아니라 실제 크기 비교)
        # forward 인자로 들어온 h, w를 100% 신뢰하여 계산합니다.
        spatial_len = h * w
        has_cls_token = (seq_len > spatial_len)
        
        # 3. [핵심] 캐시된 크기와 다르면 재계산 (Dynamic Update)
        if self.freqs_cos.shape[0] != spatial_len:
            # 기존 __init__ 로직을 여기로 가져옴 (On-the-fly calculation)
            device = t.device
            
            # --- 주파수 재생성 로직 (기존 __init__ 내용) ---
            if self.custom_freqs:
                base_freqs = self.custom_freqs
            elif self.freqs_for == 'lang':
                base_freqs = 1. / (self.theta ** (torch.arange(0, self.dim, 2, device=device)[:(self.dim // 2)].float() / self.dim))
            elif self.freqs_for == 'pixel':
                base_freqs = torch.linspace(1., self.max_freq / 2, self.dim // 2, device=device) * pi
            elif self.freqs_for == 'constant':
                base_freqs = torch.ones(self.num_freqs, device=device).float()
            
            # 2D Grid 생성 (H, W 추정)
            # # 정사각형 가정: H = W = sqrt(spatial_len)
            # h = h
            # w = w # 만약 나누어 떨어지지 않으면 직사각형 처리 필요
            
            # print('hhhhhhhhhhhhhhhhhhhhh')
            # print(h)
            # print('wwwwwwwwwwwwwwwwwwwwwwwwwww')
            # print(w)
            # H, W 그리드 만들기
            t_h = torch.arange(h, device=device).float()
            t_w = torch.arange(w, device=device).float()
            
            # print('ttttttttttttttttttttttttt')
            # print(t_h.shape)
            # print(t_w.shape)
            # einsum으로 주파수 확장
            freqs_h = torch.einsum('i, f -> i f', t_h, base_freqs)
            freqs_w = torch.einsum('i, f -> i f', t_w, base_freqs)
            
            # print('freqs_h')
            # print(freqs_h.shape)
            # print('freqs_w')
            # print(freqs_w.shape)

            # repeat (dim/2 -> dim)
            # freqs_h = repeat(freqs_h, '... n -> ... (n r)', r=2)
            # freqs_w = repeat(freqs_w, '... n -> ... (n r)', r=2)
            
            # Broadcat (H, W 결합) -> (H, W, dim)
            # print('ddddddddddddddddd')
            # print(dim)
            freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)
            # print('########################')
            # print(freqs.shape)
            # 1D로 펼치기 (H * W, dim) -> (spatial_len, dim)
            # print('self.dim')
            # print(self.dim)
            freqs = freqs.reshape(-1, self.dim)
            
            # 버퍼 업데이트 (persistent=False로 저장 안 됨 -> 로드 에러 방지!)
            self.freqs_cos = freqs.cos()
            self.freqs_sin = freqs.sin()
            # -----------------------------------------------

        # 4. 적용 (이제 크기가 무조건 맞음)
        rope_cos = self.freqs_cos
        rope_sin = self.freqs_sin
        
        # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        # print(t.shape)
        # print(rope_cos.shape)
        # print(rotate_half(t).shape)
        # print(rope_sin.shape)

        if has_cls_token:
            t_spatial = t[:, 1:, :]
            t_spatial = t_spatial * rope_cos + rotate_half(t_spatial) * rope_sin
            return torch.cat((t[:, :1, :], t_spatial), dim=1)
        else:
            return t * rope_cos + rotate_half(t) * rope_sin