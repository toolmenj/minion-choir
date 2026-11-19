"""建立小小兵人聲取樣庫：voice/<聲音代號>.flac（每隻角色一個底聲）+ voice/bank.json

做法：
1. 用 Kokoro（本機語音合成）唸出一批單音節（Bah、Nah、Bee…）。說話的母音很短，音高還會往下掉，不能直接當長音用。
2. 整體加速一點（音高和嗓音一起變尖——小小兵的聲音就是這樣來的；越高的音加速越多，但有上限）。
3. 沿著聲帶每一次振動把母音切成一小片一小片（每片兩個週期長），再照「目標音高」的間隔重新排回去：
   - 前段照原本的順序排，保留聲母和開口的過程；
   - 後段在母音最穩的那一段來回走，排成一個可以無限循環的長音。
   這樣音高固定在目標上（誤差遠小於 1 音分），嗓音的特色（共振峰）不變。

需要：pip install kokoro-onnx soundfile numpy；ffmpeg；
Kokoro 模型檔（跑過一次 `npx hyperframes tts` 就會下載到 ~/.cache/hyperframes/tts）。
用法：python tools/build_voice_bank.py            （全部重建）
      python tools/build_voice_bank.py am_onyx    （只重建指定的聲音）
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

HOME = os.path.expanduser('~')
MODEL = os.path.join(HOME, '.cache', 'hyperframes', 'tts', 'models', 'kokoro-v1.0.onnx')
VOICES = os.path.join(HOME, '.cache', 'hyperframes', 'tts', 'voices', 'voices-v1.0.bin')
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'voice')
# 九個不同的底聲（Kokoro 的男聲），網頁的 VOICE 表會指定哪隻角色用哪一個
VOICE_IDS = ['am_onyx', 'am_fenrir', 'am_puck', 'am_eric', 'am_adam', 'am_michael', 'bm_daniel', 'am_liam', 'am_echo']
SR = 44100
CYCLE = 0.45                             # 循環段長度（秒）：在母音中段來回走一趟
# (唸的字, 代號, 嘴型母音, 開口前是否閉唇)
SYLLABLES = [
    ('Bah.', 'ba', 'a', 1), ('Nah.', 'na', 'a', 0), ('Mah.', 'ma', 'a', 1), ('Pah.', 'pa', 'a', 1),
    ('Bee.', 'bi', 'i', 1), ('Mee.', 'mi', 'i', 1), ('Nee.', 'ni', 'i', 0),
    ('Boo.', 'bu', 'u', 1),
    ('Bow.', 'bo', 'o', 1), ('No.', 'no', 'o', 0),
]
MIDIS = list(range(55, 79, 3))          # G3 到 E5，每三個半音一個取樣（網頁再用播放速度微調到確切音高）


def f0_track(x, sr, lo, hi, hop=0.004, win=0.03):
    n, h = int(win * sr), int(hop * sr)
    peak = np.max(np.abs(x)) + 1e-9
    out = []
    for s in range(0, len(x) - n, h):
        seg = x[s:s + n]
        if np.sqrt(np.mean(seg ** 2)) < 0.05 * peak:
            out.append((s + n // 2, 0.0))
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, 'full')[n - 1:]
        ac = ac / (ac[0] + 1e-12)
        a, b = int(sr / hi), min(int(sr / lo), n - 2)
        k = a + int(np.argmax(ac[a:b]))
        out.append((s + n // 2, sr / k if ac[k] > 0.5 else 0.0))
    return out


def longest_run(track):
    best, cur = [], []
    for s, f in track + [(0, 0.0)]:
        if f > 0:
            cur.append((s, f))
        else:
            if len(cur) > len(best):
                best = cur
            cur = []
    return best


def resample(x, sr, speed, tmp):
    """整體加速 speed 倍（音高、嗓音一起升），回傳 44.1k 單聲道。"""
    a, b = os.path.join(tmp, 'a.wav'), os.path.join(tmp, 'b.wav')
    sf.write(a, x, sr)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', a, '-af', f'asetrate={int(round(sr * speed))},aresample={SR}', '-ac', '1', b], check=True)
    y, _ = sf.read(b)
    return y.astype(np.float64)


def pitch_marks(y, run):
    """在有聲段裡，每個聲帶週期標一個點。
    先把聲音濾到只剩基音（一條平滑的正弦狀波形）再找峰值——直接在原始波形上找，會在不同的小尖峰之間跳來跳去，
    切出來的片段就不整齊，聽起來沙沙的、像撕裂。"""
    pos = np.array([s for s, _ in run], float)
    f0s = np.array([f for _, f in run])
    a, b = int(pos[0]), int(pos[-1])
    pad = int(0.03 * SR)
    lo, hi = max(0, a - pad), min(len(y), b + pad)
    seg = y[lo:hi]
    spec = np.fft.rfft(seg)
    freq = np.fft.rfftfreq(len(seg), 1 / SR)
    keep = (freq > np.min(f0s) * 0.6) & (freq < np.max(f0s) * 1.5)
    fund = np.fft.irfft(spec * keep, len(seg))
    marks = [lo + k for k in range(1, len(fund) - 1) if fund[k] > 0 and fund[k] >= fund[k - 1] and fund[k] > fund[k + 1]]
    marks = [m for m in marks if a <= m <= b]
    # 保險：太近的重複點丟掉；中間有斷掉（間隔超過一個半週期）就只留最長的那一段
    if len(marks) < 4:
        return []
    per = np.median(np.diff(marks))
    clean = [marks[0]]
    for m in marks[1:]:
        if m - clean[-1] >= 0.5 * per:
            clean.append(m)
    best, cur = [], [clean[0]]
    for m in clean[1:]:
        if m - cur[-1] > 1.6 * per:
            best, cur = (cur if len(cur) > len(best) else best), [m]
        else:
            cur.append(m)
    return cur if len(cur) > len(best) else best


def build(y, target, region=(0.45, 0.68)):
    """回傳 (取樣, 循環起點, 循環終點, 循環段頻率)；做不出來回傳 None。"""
    base_lo, base_hi = 60, 900
    run = longest_run(f0_track(y, SR, base_lo, base_hi))
    if len(run) < 12:
        return None
    marks = pitch_marks(y, run)
    if len(marks) < 10:
        return None
    n_cyc = max(8, int(round(CYCLE * target / 2)) * 2)      # 一趟來回有幾個週期（偶數）
    L = int(round(n_cyc * SR / target))                     # 循環段長度（整數個樣本）
    T = L / n_cyc                                           # 實際的目標週期（樣本，含小數）

    def one(i):
        i = min(max(i, 1), len(marks) - 2)
        l, r = marks[i] - marks[i - 1], marks[i + 1] - marks[i]
        seg = y[marks[i] - l: marks[i] + r].copy()
        seg[:l] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(l) / l)
        seg[l:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(r) / r)
        return seg, l

    def grain(u):
        """位置 u（可以是小數）的那一片：左右各一個週期。
        u 落在兩個標記之間時，把前後兩片照比例混在一起——如果同一片重複好幾次才跳到下一片，
        音色會一格一格地變，產生粗糙的顆粒聲。"""
        i = int(np.floor(u))
        w = u - i
        g0, l0 = one(i)
        if w < 1e-6:
            return g0, l0
        g1, l1 = one(i + 1)
        L, R = max(l0, l1), max(len(g0) - l0, len(g1) - l1)
        mix = np.zeros(L + R)
        mix[L - l0: L - l0 + len(g0)] += g0 * (1 - w)
        mix[L - l1: L - l1 + len(g1)] += g1 * w
        return mix, L

    # 聲母：第一個標記之前的那一小段原封不動放在最前面
    pre = int(0.07 * SR)
    head_start = max(0, marks[0] - pre)
    head = y[head_start: marks[0]].copy()
    head[:int(0.004 * SR)] *= np.linspace(0, 1, int(0.004 * SR))
    # 加速後爆破音（b、p 的那一下）會變尖：稍微磨掉一點高頻。不能磨太多或調太小聲，不然聽不出在咬字
    head = np.convolve(head, np.ones(3) / 3, mode='same') * 0.85
    tailfade = min(len(head), marks[1] - marks[0])
    head[-tailfade:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(tailfade) / tailfade)

    # 長音只在母音最穩的一小段裡來回（約中間 18%）。範圍取太大會掃到嘴巴還在動的部分，
    # 循環起來就變成一直重複「啦啦啦」的彈舌聲
    i1 = int(len(marks) * region[0])
    i2 = min(max(i1 + 3, int(len(marks) * region[1])), len(marks) - 3)
    if i2 <= i1:
        return None
    n_on = max(2, int(round((marks[i1] - marks[0]) / T)))    # 開口段有幾個週期（照原本的時間走）
    seq = [k / n_on * i1 for k in range(n_on)]
    tri = [1 - abs(2 * c / n_cyc - 1) for c in range(n_cyc)]
    cyc = [i1 + v * (i2 - i1) for v in tri]
    seq += cyc + cyc + cyc[:6]

    out = np.zeros(len(head) + int((len(seq) + 4) * T) + 2 * SR // 50)
    out[:len(head)] += head
    p0 = len(head)
    for k, i in enumerate(seq):
        g, l = grain(i)
        centre = p0 + k * T
        start = centre - l
        s0 = int(np.floor(start))
        frac = start - s0
        if frac > 1e-6:                                     # 小數位置：用內插把這一片往後挪一點點，週期才會完全等距
            g = np.interp(np.arange(len(g) + 1) - frac, np.arange(len(g)), g, left=0, right=0)
        if s0 < 0:
            g, s0 = g[-s0:], 0
        out[s0: s0 + len(g)] += g

    ls = int(round(p0 + n_on * T)) + int(0.04 * SR)         # 循環從第一趟的 40 毫秒後開始，避開開口段的殘留
    le = ls + L
    # 把長音段的音量拉平：說話時母音的音量本來就會起伏，來回循環之後會變成一下大一下小，聽起來一斷一斷的。
    # 量每兩個週期的音量，太小聲的地方補大、太大聲的地方壓小（訊號本身是週期性的，所以修正量也是，循環接縫不受影響）
    win = max(8, int(round(2 * T)))
    env = np.sqrt(np.convolve(out ** 2, np.ones(win) / win, mode='same') + 1e-12)
    # 從母音一開口就開始拉平，不只長音段：說話的母音剛開口那 0.1–0.2 秒通常比較小聲，
    # 每換一個字就先凹一塊再大聲起來，整句聽起來就斷斷續續
    a0 = int(round(p0 + 1.5 * T))
    gain = np.ones(len(out))
    gain[a0:] = np.clip(np.median(env[ls:le]) / env[a0:], 0.5, 3.0)
    ramp = min(int(0.015 * SR), len(out) - a0)
    gain[a0:a0 + ramp] = 1 + (gain[a0:a0 + ramp] - 1) * np.linspace(0, 1, ramp)
    out = out * gain
    out = out[:le + int(0.01 * SR)]
    rms = np.sqrt(np.mean(out[ls:le] ** 2)) + 1e-9
    out *= 0.16 / rms
    return out, ls, le, SR / T, p0


def loop_evenness(out, ls, le):
    """循環段裡最小聲 ÷ 最大聲（每 6 毫秒量一次）。1 表示完全平穩；太低表示會一斷一斷。"""
    loop = out[ls:le]
    w = int(0.006 * SR)
    env = np.array([np.sqrt(np.mean(loop[k:k + w] ** 2)) for k in range(0, len(loop) - w, w // 2)])
    return float(env.min() / (env.max() + 1e-12))


def loop_pitch_error(out, ls, le, target):
    """量循環段實際的音高和目標差幾音分（品管用）。"""
    loop = out[ls:le]
    t = np.tile(loop, 3) * np.hanning(len(loop) * 3)
    sp = np.abs(np.fft.rfft(t, 1 << 18))
    fr = np.fft.rfftfreq(1 << 18, 1 / SR)
    lo, hi = np.searchsorted(fr, target * 0.9), np.searchsorted(fr, target * 1.1)
    return abs(1200 * np.log2(fr[lo + int(np.argmax(sp[lo:hi]))] / target))


def build_voice(kokoro, voice, tmp):
    chunks, index, pos = [], [], 0
    gap = np.zeros(int(0.02 * SR))
    for text, name, vowel, lips in SYLLABLES:
        x, sr = kokoro.create(text, voice=voice, speed=0.75, lang='en-us')   # 唸慢一點，母音比較長，長音的素材才夠
        x = np.asarray(x, dtype=np.float64)
        run = longest_run(f0_track(x, sr, 60, 320))
        if len(run) < 12:
            print(f'  {voice} {name}: no usable vowel, skipped')
            continue
        base = float(np.median([f for _, f in run]))
        made = 0
        for midi in MIDIS:
            target = 440 * 2 ** ((midi - 69) / 12)
            speed = float(np.clip((target / base) ** 0.55, 1.3, 2.4))
            y = resample(x, sr, speed, tmp)
            got = None
            for region in ((0.45, 0.68), (0.4, 0.72), (0.35, 0.78)):      # 先用最穩的一段；量出來音不準就放寬範圍重做
                cand = build(y, target, region)
                if cand is not None and loop_pitch_error(cand[0], cand[1], cand[2], target) < 6 and loop_evenness(cand[0], cand[1], cand[2]) > 0.55:
                    got = cand
                    break
            if got is None:                                                  # 怎麼做都不準的就不收，網頁會改用鄰近音高的取樣
                continue
            out, ls, le, f0, hd = got
            index.append({'syl': name, 'vowel': vowel, 'lips': lips, 'midi': midi, 'off': pos,
                          'len': len(out), 'ls': ls, 'le': le, 'hd': hd, 'f0': round(f0, 4)})
            chunks += [out, gap]
            pos += len(out) + len(gap)
            made += 1
        print(f'  {voice} {name}: base f0 {base:.0f} Hz, vowel {(run[-1][0] - run[0][0]) / sr * 1000:.0f} ms, {made}/{len(MIDIS)} pitches')
    bank = np.concatenate(chunks)
    peak = float(np.max(np.abs(bank)))
    if peak > 0.98:
        bank *= 0.98 / peak
    sf.write(os.path.join(OUT_DIR, voice + '.flac'), bank, SR, subtype='PCM_16')
    return {'file': voice + '.flac', 'gain': round(min(1.0, 0.98 / peak), 4), 'samples': index}, len(bank) / SR


def main():
    from kokoro_onnx import Kokoro
    kokoro = Kokoro(MODEL, VOICES)
    os.makedirs(OUT_DIR, exist_ok=True)
    meta_path = os.path.join(OUT_DIR, 'bank.json')
    meta = {'sr': SR, 'voices': {}}
    if os.path.exists(meta_path):
        old = json.load(open(meta_path, encoding='utf-8'))
        if 'voices' in old:
            meta = old
    todo = sys.argv[1:] or VOICE_IDS
    with tempfile.TemporaryDirectory() as tmp:
        for voice in todo:
            entry, secs = build_voice(kokoro, voice, tmp)
            meta['voices'][voice] = entry
            print(f'{voice}: {len(entry["samples"])} samples, {secs:.0f} s')
            json.dump(meta, open(meta_path, 'w', encoding='utf-8'))


if __name__ == '__main__':
    sys.exit(main())
