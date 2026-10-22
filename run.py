import os
from IPython import get_ipython

ipython = get_ipython()

for s in [2, 3, 4, 5]:
    os.environ['EXP_SEED'] = str(s)
    print(f"\n>>> EXP_SEED = {s} 실행 시작")
    # IPython 매직 실행 기능을 사용하여 파이썬 루프 내부에서 안전하게 실행합니다.
    ipython.run_line_magic('run', '/content/drive/MyDrive/NonlinearMemory/vt-ae/IQ_Conv1D_DVAE_with_LSTM_v7.py')
    ipython.run_line_magic('run', '/content/drive/MyDrive/NonlinearMemory/vt-ae/Transformer_DPD_conventional.py')

os.environ.pop('EXP_SEED', None)     # 평가 전에 꼭 지움 (본 평가는 기존 1회차로)
ipython.run_line_magic('run', 'memory_hpa_autoencoder_v2.py')