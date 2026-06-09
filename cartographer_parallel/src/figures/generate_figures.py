#!/usr/bin/env python3
"""Generate all report figures as PNG using matplotlib."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

plt.rcParams['font.family'] = 'Apple SD Gothic Neo'
plt.rcParams['axes.unicode_minus'] = False

OUTDIR = os.path.dirname(os.path.abspath(__file__))

def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {name}")


# ─────────────────────────────────────────────────────────────
# Fig 1: baseline_function_profile
# ─────────────────────────────────────────────────────────────
def gen_baseline_function_profile():
    funcs  = ['MakeScans', 'MakeLowCands', 'FastMatcher::Score', 'Branch', '기타']
    times  = [3.34, 12.10, 137.44, 8.52, 0.04]
    descs  = ['scan 회전/좌표 변환', 'coarse 후보 생성', '후보 점수 계산 (핵심 병목)', 'branch refine', 'bounds/grid 등']
    colors = ['#6c8ebf', '#d6b656', '#82b366', '#b85450', '#9ca3af']
    total  = sum(times)
    pcts   = [t / total * 100 for t in times]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    fig.patch.set_facecolor('white')

    n = len(funcs)
    y_pos = np.arange(n)
    ax.barh(y_pos[::-1], times, color=colors, height=0.52, left=0)

    for i, (t, p, d) in enumerate(zip(times, pcts, descs)):
        yi = n - 1 - i
        if t > 15:
            ax.text(t - 4, yi, f'{t:.2f} ms ({p:.1f}%)',
                    ha='right', va='center', fontsize=10, color='white', fontweight='bold')
        else:
            ax.text(t + 2, yi, f'{t:.2f} ms ({p:.1f}%)',
                    ha='left', va='center', fontsize=10, color='#374151')
        ax.text(1, yi - 0.35, d, ha='left', va='center', fontsize=9, color='#6b7280')

    ax.set_yticks(y_pos[::-1])
    ax.set_yticklabels(funcs[::-1], fontsize=12)
    ax.set_xlabel('시간 (ms)', fontsize=12)
    ax.set_xlim(0, 185)
    ax.set_title('기준 CUDA 함수별 실행 시간\n'
                 '조건: res0.05/depth2, 총 score 연산량 60.3M 회',
                 fontsize=14, fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', color='#e5e7eb', linewidth=0.8)
    ax.set_axisbelow(True)

    fig.text(0.5, 0.01,
             f'전체 평균 {total:.2f} ms  |  점수 계산 구간 137.44 ms (85.1%)',
             ha='center', fontsize=10, color='#374151')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, 'baseline_function_profile.png')


# ─────────────────────────────────────────────────────────────
# Fig 2a: baseline_call_structure  (CUDA 호출 구조 흐름도)
# ─────────────────────────────────────────────────────────────
def gen_baseline_call_structure():
    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.2)
    ax.axis('off')

    # Main flow boxes
    main_boxes = [
        (0.3, 1.1, 'FastMatcher\n::Score()', '전체 245 ms', '#dbeafe', '#1d4ed8'),
    ]
    # Inner repeated boxes (scan loop)
    inner_boxes = [
        (2.1, 1.1, 'scan별 반복\n(×15회)',    '각 scan마다',        '#f3f4f6', '#6b7280'),
        (4.1, 1.1, 'CUDA score 호출',         '3,721 candidates', '#fef3c7', '#d97706'),
        (6.5, 1.1, 'point loop',              '후보당 1,081 point', '#f3f4f6', '#6b7280'),
        (8.9, 1.1, 'score scatter',           'candidate에 반영',  '#f3f4f6', '#6b7280'),
    ]

    for x, y, title, sub, fc, ec in main_boxes + inner_boxes:
        rect = FancyBboxPatch((x, y), 1.65, 0.95,
                               boxstyle='round,pad=0.06',
                               facecolor=fc, edgecolor=ec, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.825, y + 0.64, title,
                ha='center', va='center', fontsize=10.5, fontweight='bold', color='#1f2937')
        ax.text(x + 0.825, y + 0.25, sub,
                ha='center', va='center', fontsize=9, color='#4b5563')

    # Arrows
    for i in range(len(main_boxes + inner_boxes) - 1):
        boxes = main_boxes + inner_boxes
        x0 = boxes[i][0] + 1.65
        x1 = boxes[i + 1][0]
        y_mid = boxes[i][1] + 0.47
        ax.annotate('', xy=(x1, y_mid), xytext=(x0, y_mid),
                    arrowprops=dict(arrowstyle='->', color='#374151', lw=1.6))

    # Loop-back annotation for "×15회 반복"
    ax.annotate('', xy=(2.1, 1.1 + 0.95), xytext=(10.55, 1.1 + 0.95),
                arrowprops=dict(arrowstyle='->', color='#374151', lw=1.2,
                                connectionstyle='arc3,rad=-0.3'))
    ax.text(6.3, 2.6, '× 15회 반복 (scan별)', ha='center', va='center',
            fontsize=9.5, color='#374151', style='italic')

    # Stats annotation
    ax.text(5.5, 0.35,
            '총 후보 55,815개  |  총 score 연산량 60.3M 회 (= 55,815 × 1,081)  |  '
            'scan별 3,721 candidates × 1,081 points',
            ha='center', va='center', fontsize=9.5, color='#6b7280')

    ax.set_title('기준 CUDA 구현의 점수 계산 호출 구조',
                 fontsize=14, fontweight='bold', pad=8)

    plt.tight_layout()
    save(fig, 'baseline_call_structure.png')


# ─────────────────────────────────────────────────────────────
# Fig 2b: baseline_call_timing  (CUDA 호출 시간 측정 결과)
# ─────────────────────────────────────────────────────────────
def gen_baseline_call_timing():
    kernel_single = 7.53           # coarse score_all() 단일 호출 (= 112.9ms / 15)
    kernel_sum    = kernel_single * 15   # 15회 CUDA event 합계 ≈ 112.95ms
    host_overhead = 137.44 - kernel_sum  # host-side 누적 overhead ≈ 24.5ms
    score_total   = 137.44

    fig, ax = plt.subplots(figsize=(7, 2.6))
    fig.patch.set_facecolor('white')

    # 상단 bar: score_all() 1회
    ax.barh(1, kernel_single, color='#6c8ebf', height=0.42)
    ax.text(kernel_single + 2, 1, f'{kernel_single:.1f} ms',
            ha='left', va='center', fontsize=10, color='#374151', fontweight='bold')

    # 하단 bar: Score() 전체 = kernel × 15 + host overhead (stacked)
    ax.barh(0, kernel_sum,    color='#6c8ebf', height=0.42, label=f'kernel × 15  ({kernel_sum:.0f} ms)')
    ax.barh(0, host_overhead, left=kernel_sum, color='#e67e22', height=0.42,
            label=f'host overhead 누적  ({host_overhead:.0f} ms)')
    ax.text(kernel_sum / 2, 0, f'{kernel_sum:.0f} ms',
            ha='center', va='center', fontsize=9, color='white')
    ax.text(kernel_sum + host_overhead / 2, 0, f'{host_overhead:.0f} ms',
            ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    ax.text(score_total + 3, 0, f'합계 {score_total:.0f} ms',
            ha='left', va='center', fontsize=9, color='#374151')

    # 15× 기준선
    ax.axvline(kernel_sum, color='#9ca3af', ls='--', lw=1, alpha=0.9)
    ax.text(kernel_sum, 1.36, f'15× = {kernel_sum:.0f} ms',
            ha='center', fontsize=8, color='#6b7280')

    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Score() 전체\n(scan 15회)', 'score_all() 1회\n(1 scan, 3,721 후보)'], fontsize=10)
    ax.set_xlabel('시간 (ms)', fontsize=10)
    ax.set_xlim(0, 200)
    ax.set_ylim(-0.5, 1.7)
    ax.set_title('scan별 반복 호출 구조의 overhead  (res0.05/depth2, 60.3M)',
                 fontsize=11, fontweight='bold', pad=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', color='#e5e7eb', lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=8.5, framealpha=0.95)

    fig.text(0.5, 0.01,
             'branch refinement 호출(1.9 ms): kernel 0.08 ms에 비해 H2D+D2H가 68%를 차지',
             ha='center', fontsize=8, color='#6b7280')

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    save(fig, 'baseline_call_timing.png')


# ─────────────────────────────────────────────────────────────
# Fig 3: score_internal_breakdown
# ─────────────────────────────────────────────────────────────
def gen_score_internal_breakdown():
    # cuda_times: CUDA score 계산 시간 (score_last_median × 15 for per-scan versions;
    #             score_last_median × 1 for batched versions ver5/ver6)
    # host_times: ScoreCoarse_median - cuda_times (dispatcher/scatter 등 CPU 측 비용)
    # 기준: CPU zero_base ScoreCoarse (median, skip>=50), CUDA logs (skip>=adaptive)
    versions   = ['기준 CUDA', 'kernel\n변경', 'buffer\n재사용', 'shmem만\n', 'bounds만\n', 'scan\nbatch', 'bounds\n감소']
    cuda_times = [112.9,  48.9, 36.5,  58.6,  87.0, 23.2, 22.0]
    host_times = [ 24.6,  17.4, 16.1,  85.1, 104.2, 14.1, 13.2]
    totals     = [c + h for c, h in zip(cuda_times, host_times)]

    x = np.arange(len(versions))
    width = 0.52

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.5),
                                    gridspec_kw={'width_ratios': [1.0, 1.0]})
    fig.patch.set_facecolor('white')

    # ── LEFT: absolute time stacked bar ──────────────────────
    ax1.bar(x, cuda_times, width, label='CUDA score 호출 시간 (막대 내부)', color='#f97316', zorder=3)
    ax1.bar(x, host_times, width, bottom=cuda_times,
            label='host-side 입력 준비·scatter 시간 (막대 위)', color='#64748b', zorder=3)

    for i, (c, h) in enumerate(zip(cuda_times, host_times)):
        total = c + h
        pct_c = c / total * 100
        pct_h = h / total * 100
        if c >= 30:
            ax1.text(x[i], c / 2, f'{c:.0f}ms\n({pct_c:.0f}%)',
                     ha='center', va='center', fontsize=9.5, color='white',
                     fontweight='bold', zorder=4)
        elif c >= 15:
            ax1.text(x[i], c / 2, f'{c:.0f}ms ({pct_c:.0f}%)',
                     ha='center', va='center', fontsize=9, color='white', zorder=4)
        else:
            ax1.text(x[i], c / 2, f'{c:.0f}ms',
                     ha='center', va='center', fontsize=8.5, color='white', zorder=4)
        ax1.text(x[i], total + 2, f'{h:.0f}ms ({pct_h:.0f}%)',
                 ha='center', va='bottom', fontsize=9, color='#475569', zorder=4)

    ax1.set_xticks(x)
    ax1.set_xticklabels(versions, fontsize=11)
    ax1.set_ylabel('시간 (ms)', fontsize=12)
    ax1.set_ylim(0, 240)
    ax1.set_title('FastMatcher::Score() 소요 시간\n'
                  'CUDA score 호출 (주황) + host-side 준비·scatter (회색)',
                  fontsize=13, fontweight='bold', pad=12)
    ax1.legend(loc='upper right', fontsize=10.5, framealpha=0.92)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', color='#e5e7eb', linewidth=0.8)
    ax1.set_axisbelow(True)

    # ── RIGHT: MatchWithWindow() time breakdown per version ───
    others_t     = 11.9   # MakeScans + Branch + 기타 (constant across versions)
    make_cands_t = 12.1   # MakeLowCands (constant across versions)
    # 기준 CPU: CPU zero_base ScoreCoarse median (skip>=50) = 249.8ms, no dispatcher
    # CUDA: ScoreCoarse median (skip>=50) decomposed into CUDA score + host dispatcher
    score_breakdown = {
        '기준\nCPU':           [249.8,   0.0],
        '기준\nCUDA':          [112.9,  24.6],
        'kernel\n변경':        [ 48.9,  17.4],
        'buffer\n재사용':      [ 36.5,  16.1],
        'shmem만\n': [ 58.6,  85.1],
        'bounds만\n':[ 87.0, 104.2],
        'scan\nbatch':         [ 23.2,  14.1],
        'bounds\n감소':        [ 22.0,  13.2],
    }
    score_keys  = list(score_breakdown.keys())
    compute_t   = [score_breakdown[k][0] for k in score_keys]
    host_t      = [score_breakdown[k][1] for k in score_keys]
    xb          = np.arange(len(score_keys))

    bottom_base  = np.array([others_t] * len(score_keys))
    bottom_cands = bottom_base + make_cands_t
    bottom_host  = bottom_cands + np.array(host_t)

    ax2.bar(xb, [others_t] * len(score_keys),
            color='#e5e7eb', label='MakeScans + Branch + 기타', zorder=3, width=0.52)
    ax2.bar(xb, [make_cands_t] * len(score_keys), bottom=bottom_base,
            color='#6c8ebf', label='MakeLowCands', zorder=3, width=0.52)
    ax2.bar(xb, host_t, bottom=bottom_cands,
            color='#64748b', label='host-side 입력·결과 반영 비용 (dispatcher)', zorder=3, width=0.52)
    ax2.bar(xb, compute_t, bottom=bottom_host,
            color='#f97316', label='CUDA/CPU score 계산', zorder=3, width=0.52)

    for i, (c, h) in enumerate(zip(compute_t, host_t)):
        total = others_t + make_cands_t + h + c
        ax2.text(i, total + 6, f'{total:.0f} ms',
                 ha='center', va='bottom', fontsize=8.5, color='#374151', fontweight='bold')

    ax2.set_xticks(xb)
    ax2.set_xticklabels(score_keys, fontsize=10)
    ax2.set_ylabel('시간 (ms)', fontsize=12)
    ax2.set_ylim(0, 340)
    ax2.set_title('res0.05/depth2 기준 MatchWithWindow() 소요 시간 분해\n'
                  '최적화 단계별 FastMatcher::Score() 내부 구조 변화',
                  fontsize=13, fontweight='bold', pad=12)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', color='#e5e7eb', linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.legend(loc='upper right', fontsize=9.5, framealpha=0.92)

    plt.tight_layout(w_pad=3.0)
    save(fig, 'score_internal_breakdown.png')


# ─────────────────────────────────────────────────────────────
# Fig 4: batch_call_breakdown
# ─────────────────────────────────────────────────────────────
def gen_batch_call_breakdown():
    # ── RIGHT panel data: workload별 비중 (ver6 nvprof 실측) ──────
    # 전 조건 ver6 batched profile 안정 구간 평균 실측값
    # 60.3M (calls=1700-2200): h2d=1.07ms, kernel=19.54ms, d2h=0.35ms, host=0.48ms, total=21.44ms
    # 237M  (calls=300-700):   h2d=1.71ms, kernel=55.05ms, d2h=3.25ms, host=1.30ms, total=61.31ms
    # 1.47B (calls=100):       h2d=9.30ms, kernel=450.37ms, d2h=14.35ms, host=4.76ms, total=478.78ms
    # 5.68B (calls=5-10):      h2d=89.8ms, kernel=1934ms, d2h=27.5ms, host=29.0ms, total=2080ms
    workload_labels = ['60.3M', '237M', '1.47B', '5.68B']
    h2d_pct  = [5.0, 2.8, 1.9, 4.3]
    kern_pct = [91.1, 89.8, 94.1, 93.0]
    d2h_pct  = [1.6, 5.3, 3.0, 1.3]
    host_pct = [2.3, 2.1, 1.0, 1.4]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.5),
                                    gridspec_kw={'width_ratios': [1.0, 1.0]})
    fig.patch.set_facecolor('white')

    # ─── LEFT: 4개 100% 비율 stacked bar ─────────────────────
    # Group 1 (bars 0,1): MatchWithWindow 내부 구성 비율
    #   기준 CUDA MatchWithWindow ≈103ms (speedup 역산, nvprof 제외)
    #   최종 batch MatchWithWindow ≈48.6ms (single_depth2 로그 실측)
    # Group 2 (bars 2.5,3.5): 개별 score 호출 내부 구성
    #   기준 CUDA 단일 scan 호출 ≈14.5ms (ver1 nvprof)
    #   최종 batch 호출 ≈19.5ms (ver5 실측)

    bar_positions = [0, 1, 2.5, 3.5]
    bar_labels = [
        '기준 CUDA\n(MatchWithWindow)',
        'bounds 감소\n(MatchWithWindow)',
        '기준 CUDA\n(단일 scan 호출)',
        'bounds 감소\n(batch 호출)',
    ]
    bar_w = 0.65

    # Group 1 segments: 기타 / MakeLowCands / dispatcher / CUDA score
    g1_colors = ['#d1d5db', '#3b82f6', '#64748b', '#f97316']
    g1_labels_legend = ['기타 (MakeScans+Branch)', 'MakeLowCands', 'dispatcher (host prep)', 'CUDA score']
    g1_data = [
        [0.8,  1.5,  37.9, 59.8],   # 기준 CUDA MatchWithWindow
        [9.3,  24.9, 25.7, 40.1],   # 최종 batch MatchWithWindow
    ]

    # Group 2 segments: H2D / CUDA kernel / D2H / host overhead
    # 기준 CUDA 단일 scan 호출: ver1 nvprof 실측 기반 (총 ≈14.5ms)
    # 최종 batch 호출: ver6 60.3M 실측 (h2d=5.0%, kernel=91.1%, d2h=1.6%, host=2.3%)
    g2_colors = ['#38bdf8', '#2563eb', '#22c55e', '#9ca3af']
    g2_labels_legend = ['Memcpy H2D', 'CUDA kernel', 'Memcpy D2H', 'host/overhead']
    g2_data = [
        [3.5, 86.2, 3.5, 6.8],   # 기준 CUDA 단일 scan 호출
        [5.0, 91.1, 1.6, 2.3],   # 최종 batch 호출 (ver6 실측)
    ]

    def draw_stacked_bars(ax, positions, data, colors, legend_labels, group_idx):
        bottoms = [0.0] * len(positions)
        for seg_i, (color, seg_label) in enumerate(zip(colors, legend_labels)):
            for bar_i, pos in enumerate(positions):
                val = data[bar_i][seg_i]
                bot = bottoms[bar_i]
                ax.bar(pos, val, bottom=bot, color=color, width=bar_w, zorder=3,
                       label=seg_label if bar_i == 0 else '_nolegend_')
                mid_y = bot + val / 2
                if val >= 5.0:
                    ax.text(pos, mid_y, f'{val:.1f}%',
                            ha='center', va='center', fontsize=9,
                            color='white' if color in ('#f97316', '#2563eb', '#64748b') else '#0f172a',
                            fontweight='bold', zorder=4)
                elif val >= 2.0:
                    ax.text(pos, mid_y, f'{val:.1f}%',
                            ha='center', va='center', fontsize=8,
                            color='white' if color in ('#f97316', '#2563eb', '#64748b') else '#0f172a',
                            zorder=4)
                bottoms[bar_i] += val

    # Draw group 1
    draw_stacked_bars(ax1, bar_positions[:2], g1_data, g1_colors, g1_labels_legend, 0)
    # Draw group 2
    draw_stacked_bars(ax1, bar_positions[2:], g2_data, g2_colors, g2_labels_legend, 1)

    # Group separator line
    ax1.axvline(x=1.75, color='#9ca3af', linewidth=1.2, linestyle='--', zorder=2)

    ax1.set_xticks(bar_positions)
    ax1.set_xticklabels(bar_labels, fontsize=9.5)
    ax1.set_xlim(-0.6, 4.1)
    ax1.set_ylabel('비중 (%)', fontsize=12)
    ax1.set_ylim(0, 130)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.set_title('기준 CUDA vs bounds 감소: 함수 구성 비율 비교',
                  fontsize=12.5, fontweight='bold', pad=12)

    # 각 그룹 상단에 개별 제목 박스
    ax1.text(0.5, 117, 'MatchWithWindow() 구성 비율', ha='center', va='center',
             fontsize=9, color='white', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.35', facecolor='#475569', edgecolor='none'))
    ax1.text(3.0, 117, 'score 호출 내부 구성 비율', ha='center', va='center',
             fontsize=9, color='white', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.35', facecolor='#475569', edgecolor='none'))

    # 두 개의 별도 legend: Group 1 (MatchWithWindow), Group 2 (score 호출 내부)
    import matplotlib.legend as mlegend
    patches_g1 = [mpatches.Patch(color=c, label=l) for c, l in zip(g1_colors, g1_labels_legend)]
    patches_g2 = [mpatches.Patch(color=c, label=l) for c, l in zip(g2_colors, g2_labels_legend)]
    leg1 = ax1.legend(handles=patches_g1,
                      title='MatchWithWindow 분해', title_fontsize=8,
                      loc='upper left', bbox_to_anchor=(-0.05, 1.0),
                      fontsize=8.0, framealpha=0.92, ncol=1)
    ax1.add_artist(leg1)
    ax1.legend(handles=patches_g2,
               title='score 호출 내부', title_fontsize=8,
               loc='upper left', bbox_to_anchor=(0.56, 1.0),
               fontsize=8.0, framealpha=0.92, ncol=1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', color='#e5e7eb', linewidth=0.8)
    ax1.set_axisbelow(True)

    # ─── RIGHT: 100% stacked bar across workloads (ver6 실측) ─
    xb = np.arange(len(workload_labels))
    bar_w_r = 0.55
    right_colors = ['#38bdf8', '#2563eb', '#22c55e', '#64748b']
    right_cats = ['Memcpy H2D', 'CUDA kernel', 'Memcpy D2H', 'host overhead']

    b0 = np.zeros(len(workload_labels))
    for comp_pct, c, cat in zip(
        [h2d_pct, kern_pct, d2h_pct, host_pct],
        right_colors, right_cats
    ):
        vals = np.array(comp_pct)
        ax2.bar(xb, vals, bottom=b0, color=c, label=cat, width=bar_w_r, zorder=3)
        for i, (v, bot) in enumerate(zip(vals, b0)):
            mid_y = bot + v / 2
            if v >= 1.5:
                ax2.text(i, mid_y, f'{v:.1f}%',
                         ha='center', va='center', fontsize=9,
                         color='white' if cat == 'CUDA kernel' else '#0f172a',
                         fontweight='bold' if cat == 'CUDA kernel' else 'normal',
                         zorder=4)
        b0 = b0 + vals

    ax2.set_xticks(xb)
    ax2.set_xticklabels(workload_labels, fontsize=11)
    ax2.set_ylabel('비중 (%)', fontsize=12)
    ax2.set_ylim(0, 108)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.set_title('workload 증가에 따른 batch 호출 내부 비중 변화\n'
                  '(bounds 감소 실측 기반)',
                  fontsize=13, fontweight='bold', pad=12)
    ax2.legend(loc='lower right', fontsize=10, framealpha=0.92)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', color='#e5e7eb', linewidth=0.8)
    ax2.set_axisbelow(True)

    plt.tight_layout(w_pad=3.0)
    save(fig, 'batch_call_breakdown.png')


# ─────────────────────────────────────────────────────────────
# Fig 5: workload_speedup_comparison  (기준 CPU = 1.0x)
# Single panel: speedup across workloads (right panel moved to score_internal_breakdown)
# ─────────────────────────────────────────────────────────────
def gen_workload_speedup_comparison():
    workload_labels = ['60.3M', '237M', '1.47B', '5.68B']
    x = np.arange(len(workload_labels))

    # Speedup vs CPU baseline: CPU zero_base ScoreCoarse (median, calls>=adaptive skip)
    # divided by CUDA/CPU-OMP version ScoreCoarse (median, calls>=adaptive skip)
    # skip thresholds: 60.3M>=50, 237M>=20, 1.47B>=5, 5.68B>=3
    # CPU OpenMP SIMD: score_all time (covers all scans in one call), 4 threads
    spd = {
        '기준 CPU':              np.array([1.000, 1.000, 1.000, 1.000]),
        'CPU OpenMP':           np.array([3.077, 3.248, 3.395, 3.392]),  # openmp_simd
        '기준 CUDA':             np.array([1.818, 2.181, 6.714, 6.911]),  # baseline
        'kernel 변경':           np.array([3.773, 4.240, 6.905, 7.025]),  # ver2
        'buffer 재사용':         np.array([4.749, 7.543, 5.597, 5.871]),  # ver4
        'shmem만':    np.array([1.739, 1.864, 1.550, 1.633]),  # ver1
        'bounds만':   np.array([1.307, 1.574, 1.431, 1.566]),  # ver3
        'scan batch':           np.array([6.704, 8.510, 7.335, 6.780]),  # ver5
        'bounds 감소':           np.array([7.113, 9.255, 6.572, 6.072]),  # ver6 (최종)
    }

    styles = {
        '기준 CPU':             dict(color='#111827', ls='--', lw=2.2, marker='o', ms=6),
        'CPU OpenMP':          dict(color='#10b981', ls='--', lw=2.0, marker='o', ms=6),
        '기준 CUDA':            dict(color='#6b7280', ls=':',  lw=2.2, marker='o', ms=6),
        'kernel 변경':          dict(color='#f97316', ls='-',  lw=2.0, marker='s', ms=6),
        'buffer 재사용':        dict(color='#0ea5e9', ls='-',  lw=2.0, marker='s', ms=6),
        'shmem만':   dict(color='#a3e635', ls=':',  lw=1.8, marker='^', ms=6),
        'bounds만':  dict(color='#fb923c', ls=':',  lw=1.8, marker='^', ms=6),
        'scan batch':          dict(color='#8b5cf6', ls='-',  lw=2.0, marker='s', ms=6),
        'bounds 감소':          dict(color='#dc2626', ls='--', lw=2.5, marker='D', ms=7),
    }

    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    fig.patch.set_facecolor('white')

    for name, vals in spd.items():
        s = styles[name]
        ax.plot(x, vals, color=s['color'], ls=s['ls'], lw=s['lw'],
                marker=s['marker'], markersize=s['ms'],
                markerfacecolor='white', markeredgewidth=1.8, label=name)

    ax.axhline(1.0, color='#111827', lw=1.0, ls='-', alpha=0.25, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(workload_labels, fontsize=12)
    ax.set_xlabel('총 score 연산량', fontsize=13)
    ax.set_ylabel('speedup (기준 CPU = 1.0×)', fontsize=13)
    ax.set_ylim(0, 11)
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ax.set_title('workload 증가에 따른 기준 CPU 대비 성능 향상\n'
                 '1× = 기준 CPU와 동일  ·  bounds 감소는 점선',
                 fontsize=14, fontweight='bold', pad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(color='#e5e7eb', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize=11, framealpha=0.92, borderaxespad=0)

    plt.tight_layout()
    save(fig, 'workload_speedup_comparison.png')


if __name__ == '__main__':
    gen_baseline_function_profile()
    gen_baseline_call_structure()
    gen_baseline_call_timing()
    gen_score_internal_breakdown()
    gen_batch_call_breakdown()
    gen_workload_speedup_comparison()
    print("All figures generated.")
