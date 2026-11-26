# -*- coding: utf-8 -*-
"""精修全部动作帧（v2）：
   1) 只从边界泛洪去白底，再把误抠的内部透明洞用原图颜色补回（修“脸/身上出现黑块”）；
   2) 调色板 + 块众数降采样（每个输出像素取源块内出现最多的颜色），不用均值——均值会把深色发丝
      和肤色搅成黑块，这一版保住原画的像素格；
   3) 全部帧用同一缩放系数（源图 1536 → 124），保证动作之间人物大小一致；
   4) 清 <4px 孤立碎点，贴进 160×128 帧框（基线 118）。
"""
import os, io, json, urllib.request, numpy as np
from PIL import Image
from collections import deque, Counter

BASE = "https://aka.doubaocdn.com/s/"
GROUPS = {
    "idle":  ["bvCAkIQ0Zy","BfVwYu4NU3","wMg1CTM6h5"],
    "walk":  ["FuTLRPQCoq","VzGfL7NMRX","UsqX7RsJma","iXI9xEKmS8"],
    "run":   ["68V1G2gWAi","0Y94tNOmVA","wT9VetyVLF"],
    "jump":  ["dbKBgz9GFQ","gVYzLpfNf2","SAKDDKWS62"],
    "dash":  ["kSc4A4vESv","e6WVqZNmzT","3tCaKiO2J1"],
    "attack":["paCIf48wnT","B26o8qnxx4","9e6VvUXMXV","xVPYvU7Vdf"],
    "down":  ["NfvKZYtmMm","CbhDxtBwez","quJTfKs3WF","76RIhxHbAl"],
    "hit":   ["cy8sP8YDE0","Ka6DZedzim","nDY8RJqrkX","UD05sdLPgk"],
}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refined")
FW, FH, BASELINE = 160, 128, 118
CONTACT_GROUPS = False   # 由调用方按组置 True（down/hit）
SRC = 1536
SCALE = 0.0695          # 全局固定缩放：站姿角色约 97px 高，各姿势按真实比例，不再逐帧拉满

def regions(mask, conn=4):
    hh, ww = mask.shape; seen = np.zeros((hh, ww), bool); out = []
    nbrs = ((1,0),(-1,0),(0,1),(0,-1)) if conn == 4 else \
           ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1))
    for y in range(hh):
        for x in range(ww):
            if mask[y, x] and not seen[y, x]:
                q = deque([(y, x)]); seen[y, x] = True; pts = []; touch = False
                while q:
                    cy, cx = q.popleft(); pts.append((cy, cx))
                    if cy in (0, hh-1) or cx in (0, ww-1): touch = True
                    for dy, dx in nbrs:
                        ny, nx = cy+dy, cx+dx
                        if 0 <= ny < hh and 0 <= nx < ww and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; q.append((ny, nx))
                out.append((pts, touch))
    return out

def remove_bg(a):
    h, w, _ = a.shape
    rgb = a[:, :, :3].astype(int); lo = rgb.min(2); hi = rgb.max(2); sp = hi - lo
    bg = ((lo > 172) & (sp < 42)) | ((lo > 108) & (sp < 18))
    vis = np.zeros((h, w), bool); q = deque()
    for x in range(w):
        for y in (0, h-1):
            if bg[y, x] and not vis[y, x]: q.append((y, x)); vis[y, x] = True
    for y in range(h):
        for x in (0, w-1):
            if bg[y, x] and not vis[y, x]: q.append((y, x)); vis[y, x] = True
    while q:
        y, x = q.popleft()
        for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
            ny, nx = y+dy, x+dx
            if 0 <= ny < h and 0 <= nx < w and bg[ny, nx] and not vis[ny, nx]:
                vis[ny, nx] = True; q.append((ny, nx))
    orig = a.copy()
    a[vis, 3] = 0
    # 回填内部洞：被误判成背景、但与画面边缘不连通的区域
    holes = (a[:, :, 3] < 10) & ~vis
    for pts, touch in regions(holes, 4):
        if touch: continue
        ys = np.array([p[0] for p in pts]); xs = np.array([p[1] for p in pts])
        a[ys, xs] = orig[ys, xs]; a[ys, xs, 3] = 255
    return a

def drop_low_white(a):
    """删掉被扬尘/速度线围住的白色背景残块：近白、面积大、重心在画面下部的连通块"""
    import scipy.ndimage as nd
    rgb = a[:, :, :3].astype(int)
    lo = rgb.min(2); sp = rgb.max(2) - lo
    white = (a[:, :, 3] > 0) & (lo > 168) & (sp < 48)
    lab, n = nd.label(white, structure=np.ones((3, 3)))
    h, w = white.shape
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(ys) < 260: continue
        if ys.mean() < 0.74 * h: continue          # 只动下半部（围裙在上半身的不会被误伤）
        if xs.max() - xs.min() > 0.55 * w: continue  # 太宽的大概率是围裙
        a[ys, xs, 3] = 0
    return a

def block_mode(a, out_h):
    """块众数降采样：每个输出像素取源块内出现最多的颜色桶，并用该桶均值还原色相。
       不用均值/全局调色板——前者会把深发丝与肤色搅成黑块，后者会整体发白。"""
    ys, xs = np.where(a[:, :, 3] > 10)
    if len(xs) == 0: return None
    sub = a[ys.min():ys.max()+1, xs.min():xs.max()+1]
    h, w = sub.shape[:2]; ow = max(1, round(w * out_h / h)); oh = out_h
    oy0 = (np.arange(oh+1) * h / oh).astype(int); ox0 = (np.arange(ow+1) * w / ow).astype(int)
    key = (sub[:, :, 0] >> 4).astype(np.int32) * 1024 + (sub[:, :, 1] >> 4) * 32 + (sub[:, :, 2] >> 4)
    out = np.zeros((oh, ow, 4), np.uint8)
    for oy in range(oh):
        for ox in range(ow):
            ys0, ys1, xs0, xs1 = oy0[oy], oy0[oy+1], ox0[ox], ox0[ox+1]
            ba = sub[ys0:ys1, xs0:xs1, 3]
            if ba.size == 0: continue
            if (ba > 40).mean() < 0.34: continue          # 大半是背景 → 这一格透明
            bk = key[ys0:ys1, xs0:xs1]
            m = ba > 100
            vals = bk[m] if m.any() else bk
            if vals.size == 0: continue
            top = Counter(vals.ravel().tolist()).most_common(1)[0][0]
            sel = (bk == top) & (ba > 40)
            out[oy, ox, :3] = sub[ys0:ys1, xs0:xs1, :3][sel].mean(0).round().astype(np.uint8)
            out[oy, ox, 3] = 255
    return Image.fromarray(out, "RGBA")

def grade(arr):
    """暗部提亮：纯黑布/发在深色场景里会看成“洞”，抬到深紫褐并保留层次"""
    rgb = arr[:, :, :3].astype(np.float32); m = rgb.mean(2)
    k = m < 28
    base = np.array([36, 25, 33], np.float32)
    rgb[k] = base + (rgb[k] - m[k, None]) * 0.45
    mid = (m >= 28) & (m < 72)
    rgb[mid] *= 1.14
    arr[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return arr

def refine(img):
    a = remove_bg(np.array(img.convert("RGBA")))
    ys, xs = np.where(a[:, :, 3] > 10)
    bh = ys.max() - ys.min() + 1
    whole = block_mode(a, max(1, int(round(bh * SCALE))))
    if whole is None: return None
    arr = grade(drop_low_white(np.array(whole)))
    # 清孤立碎点
    al = arr[:, :, 3] > 10
    for pts, touch in regions(al, 8):
        if len(pts) < 4:
            for cy, cx in pts: arr[cy, cx, 3] = 0
    ys, xs = np.where(arr[:, :, 3] > 10)
    if len(xs) == 0: return None
    crop = Image.fromarray(arr[ys.min():ys.max()+1, xs.min():xs.max()+1])
    cv = Image.new("RGBA", (FW, FH), (0, 0, 0, 0))
    # 躺/坐/跪姿的尾巴尖、翼尖会比身体接地点更低几像素；按包围盒底对齐会让身体悬空。
    # 这些组改用「身体接触行」= 不透明宽度>=8 的最低行 作为基线，细尖端允许压到基线下方。
    if CONTACT_GROUPS:
        ca = np.array(crop)[:, :, 3] > 20
        widths = ca.sum(1)
        rows = np.where(widths >= 8)[0]
        contact = int(rows.max()) + 1 if len(rows) else crop.height
    else:
        contact = crop.height
    cv.paste(crop, (80 - crop.width//2, BASELINE - contact), crop)
    return cv

def audit(im):
    a = np.array(im); tr = a[:, :, 3] < 10
    n = 0
    for pts, touch in regions(tr, 4):
        if not touch and len(pts) > 6: n += 1
    return n

def main():
    os.makedirs(OUT, exist_ok=True); man = {}
    for g, ids in GROUPS.items():
        frames = []
        for i, uid in enumerate(ids, 1):
            d = urllib.request.urlopen(BASE + uid, timeout=25).read()
            cv = refine(Image.open(io.BytesIO(d)))
            p = os.path.join(OUT, f"{g}{i}.png"); cv.save(p); frames.append(p)
            print(f"{g}{i} {cv.size} holes={audit(cv)}")
        man[g] = frames
    json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), ensure_ascii=False, indent=1)
    rows = []
    for g, frames in man.items():
        row = Image.new("RGBA", (FW*len(frames), FH), (40, 48, 60, 255))
        for i, f in enumerate(frames): row.alpha_composite(Image.open(f), (i*FW, 0))
        rows.append(row)
    sheet = Image.new("RGBA", (max(r.width for r in rows), FH*len(rows)), (24, 27, 33, 255))
    for i, r in enumerate(rows): sheet.alpha_composite(r, (0, i*FH))
    sheet.resize((sheet.width*2, sheet.height*2), Image.NEAREST).save(os.path.join(OUT, "_contact.png"))
    print("done")

if __name__ == "__main__":
    main()
