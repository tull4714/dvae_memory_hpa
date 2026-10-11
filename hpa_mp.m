function [y, A_sat] = hpa_mp(x, C, IBO_dB, model, ref_power)
% HPA_MP  메모리 다항식 HPA 모델 (파이썬 dvae_curriculum.polynomial_np 와 동일)
%   [y, A_sat] = hpa_mp(x, C, IBO_dB, model, ref_power)
%   x         : 복소 기저대역 입력 (열벡터, 물리적 시간 순서)
%   C         : 3x3 계수 행렬 (행: k = 1,3,5 / 열: q = 0,1,2), Ding 등의 원 계수
%               내부에서 c10으로 나누어 소신호 이득을 1로 정규화
%   IBO_dB    : 입력 백오프 [dB], 평균 입력 전력이 포화 입력 전력보다 IBO_dB 낮음
%   model     : 'standard' - 논문 식 (2)의 표준 메모리 다항식 (기본)
%               'code'     - 이전 코드의 변형 (메모리 항에 지연된 진폭만 사용)
%   ref_power : PA 기준 전력 (원신호 평균 전력). 생략하면 x 자신의 평균 전력
%               DPD 출력을 넣을 때는 원신호의 평균 전력을 넘겨 PA를 고정해야 함
%   A_sat     : 포화 입력 진폭 (메모리 없는 AM/AM이 가장 평평한 점)

if nargin < 4 || isempty(model), model = 'standard'; end
x  = x(:);
if nargin < 5 || isempty(ref_power), ref_power = mean(abs(x).^2); end
Cn = C / C(1,1);                                     % 소신호 이득 1로 정규화
korder = [1 3 5];

% 포화 입력 진폭: |F0(a)| 의 기울기가 가장 작은 점
a  = linspace(1e-4, 1, 100000);
f0 = Cn(1,1)*a + Cn(2,1)*a.^3 + Cn(3,1)*a.^5;
[~, i] = min(gradient(abs(f0), a));
A_sat = a(i);

s = sqrt(ref_power * 10^(IBO_dB/10)) / A_sat;        % 평균 입력 전력 = 포화 입력 전력 - IBO
u = x / s;
m = abs(u);
over = m > A_sat;
u(over) = u(over) ./ m(over) * A_sat;                % 진폭 제한기 (포화)

acc = zeros(size(u));
for q = 0:size(Cn,2)-1
    if q == 0, ud = u; else, ud = [zeros(q,1); u(1:end-q)]; end
    for ki = 1:size(Cn,1)
        k = korder(ki);
        if strcmp(model, 'code')
            acc = acc + Cn(ki,q+1) * abs(ud).^k;
        else
            acc = acc + Cn(ki,q+1) * ud .* abs(ud).^(k-1);
        end
    end
end
if strcmp(model, 'code'), acc = acc .* exp(1j*angle(u)); end
y = acc * s;
end
