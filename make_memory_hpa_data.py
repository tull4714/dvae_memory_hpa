"""
학습 데이터 생성: 16QAM OFDM 원신호 → HPA(표준 memory polynomial) → CSV 저장

[변경 사항]
1) 시간 신호를 물리적 전송 순서(심볼 단위로 256샘플씩)로 펼침  (reshape order='F')
   이전에는 행 우선 순서라 스트림의 이웃 샘플이 '다음 OFDM 심볼의 같은 위치 샘플'이었음.
2) HPA는 dvae_curriculum.py의 polynomial_np (표준 memory polynomial) 하나만 사용.
3) 학습 데이터와 평가 데이터가 같은 신호가 되지 않도록 난수 시드를 분리 (학습 SEED = 1, 평가 SEED = 2).
4) BER 확인 루프는 평가 파일과 중복이라 제거하고, 대신 HPA 왜곡 수준만 출력.
5) PA 계수와 IBO는 dvae_curriculum.py (Ding 계수, 포화 기준 IBO)에서 가져옴.
"""
import os
import sys
import numpy as np
import importlib.util, shutil

# ============================================================
# 경로와 파라미터
# ============================================================
BASE_DIR  = '/content/drive/MyDrive'
MODEL_DIR = os.path.join(BASE_DIR, 'NonlinearMemory')

Fs, Fd   = 4, 1          # 오버샘플링 4배
N        = 64            # 부반송파 수
Block    = 100000        # OFDM 심볼 수
M        = 16            # 16QAM
D        = N * Block
SEED     = 1             # 학습 데이터용 시드 (평가 파일은 2를 사용)

# HPA 모델 로드 (dvae_curriculum.py — 학습/평가와 같은 정의를 공유)
_path = os.path.join(MODEL_DIR, 'dvae_curriculum.py')
for _key in list(sys.modules.keys()):
    if 'dvae_curriculum' in _key:
        del sys.modules[_key]
_spec = importlib.util.spec_from_file_location("dvae_curriculum", _path)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
polynomial_np = _mod.polynomial_np
back_off      = _mod.PA_IBO_DB      # IBO [dB] (포화 입력 전력 기준, dvae_curriculum.py에서 한 번만 정의)


# ============================================================
# 함수
# ============================================================
def gen_mapping(M, D, seed):
    rng = np.random.RandomState(seed)
    rand_I = rng.rand(1, D)
    rand_Q = rng.rand(1, D)
    if M == 2:
        s = 2 * np.fix(rand_I * 2) - 1
    elif M == 4:
        s = (2 * np.fix(rand_I * 2) - 1) + 1j * (2 * np.fix(rand_Q * 2) - 1)
    elif M == 16:
        s = ((-2 * np.fix(rand_I * 4) + 3) + 1j * (-2 * np.fix(rand_Q * 4) + 3)) / np.sqrt(10.0)
    return s

def ofdm_mod(b, N, Fs, Fd):
    """심볼 → 오버샘플된 시간 신호 (물리적 전송 순서, 열벡터)"""
    sp = b.reshape(N, -1)                                   # (부반송파, OFDM 심볼)
    Nfft = int(Fs / Fd * N)
    x1 = np.concatenate((sp[:N // 2], np.zeros((Nfft - N, sp.shape[1]), dtype=complex), sp[N // 2:]))
    t = N * Fs / Fd * np.fft.ifft(x1, axis=0)               # 열 하나 = OFDM 심볼 하나
    return t.reshape(-1, 1, order='F')                      # 심볼 단위로 이어 붙임

def ofdm_demod(rx, N, Fs, Fd):
    """시간 신호 → 데이터 부반송파 심볼 (ofdm_mod와 같은 순서)"""
    Nfft = int(Fs / Fd * N)
    sp = ((Fd / Fs) / N * rx).reshape(Nfft, -1, order='F')
    f = np.fft.fft(sp, axis=0)
    return np.concatenate((f[:N // 2], f[Nfft - N // 2:])).reshape(1, -1)


# ============================================================
# 신호 생성 → HPA → 저장
# ============================================================
b        = gen_mapping(M, D, SEED)
IQ_out_r = ofdm_mod(b, N, Fs, Fd)
hpa_out  = polynomial_np(IQ_out_r, back_off)             # 원신호만 넣으므로 기준 전력 = 자기 전력

# 확인용 출력
papr = 10 * np.log10(np.max(np.abs(IQ_out_r) ** 2) / np.mean(np.abs(IQ_out_r) ** 2))
evm_t  = np.linalg.norm(hpa_out - IQ_out_r) / np.linalg.norm(IQ_out_r) * 100
tx_sym = b.reshape(1, -1)
evm_ib = np.sqrt(np.mean(np.abs(ofdm_demod(hpa_out, N, Fs, Fd) - tx_sym) ** 2) / np.mean(np.abs(tx_sym) ** 2)) * 100
evm_0  = np.sqrt(np.mean(np.abs(ofdm_demod(IQ_out_r, N, Fs, Fd) - tx_sym) ** 2) / np.mean(np.abs(tx_sym) ** 2)) * 100
adj = np.abs(np.vdot(IQ_out_r[:-1], IQ_out_r[1:])) / np.vdot(IQ_out_r, IQ_out_r).real
print(f"샘플 수 {IQ_out_r.size}, PAPR {papr:.1f} dB, 인접 샘플 상관 {adj:.3f} (물리 순서면 약 0.9)")
print(f"복조 확인 (0이어야 함): {evm_0:.2e} %")
print(f"IBO {back_off} dB (포화 기준), 포화 초과 샘플 {100*np.mean(np.abs(IQ_out_r)/np.sqrt(np.mean(np.abs(IQ_out_r)**2)*10**(back_off/10))*_mod.A_SAT > _mod.A_SAT):.3f} %")
print(f"HPA EVM: 시간 영역 {evm_t:.2f} %, 대역 내(복조 후) {evm_ib:.2f} % (이득 정규화 전)")
_tx = b.reshape(-1); _rx = ofdm_demod(hpa_out, N, Fs, Fd).reshape(-1)
_g = np.vdot(_tx, _rx) / np.vdot(_tx, _tx)
print(f"PA {_mod.PA_ORDER}차, A_SAT {_mod.A_SAT:.3f}: 이득 정규화 대역 내 EVM {np.sqrt(np.mean(np.abs(_rx/_g - _tx)**2)/np.mean(np.abs(_tx)**2))*100:.2f} %, |g| {abs(_g):.4f}")
print("※ 학습 데이터(input_target)는 PA 차수와 무관하므로 PA_ORDER만 바꿀 때는 데이터를 다시 만들 필요가 없다")

def save_column(path, v):
    np.savetxt(path, v, fmt='%.12e')
    print(f"저장: {path}")

save_column(os.path.join(BASE_DIR, 'input_target_I.csv'), np.real(IQ_out_r))   # 원신호 (학습 입력)
save_column(os.path.join(BASE_DIR, 'input_target_Q.csv'), np.imag(IQ_out_r))
save_column(os.path.join(BASE_DIR, 'input_iq_I.csv'), np.real(hpa_out))        # HPA 출력 (참고용)
save_column(os.path.join(BASE_DIR, 'input_iq_Q.csv'), np.imag(hpa_out))
