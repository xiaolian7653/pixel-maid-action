# -*- coding: utf-8 -*-
"""重生成待机/奔跑两组：新源图（无速度线/扬尘），按目标高度归一"""
import io, os, json, urllib.request, importlib.util, numpy as np
from PIL import Image
spec=importlib.util.spec_from_file_location("r", os.path.join(os.path.dirname(__file__),"refine_all.py"))
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

NEW={
 "idle": (["URrfOrdjlY","PM7aEcjHhV","P6ZltFnKKT"], 100),  # 睁眼/呼气/眨眼
 "run":  (["LRgjIcMdA0","NXjJNVms9J","KO78pgVN1s","pbFCHYbbdS"], 95),
 "walk": (["OEtVkGPjxw","CHVlUSoGbz","exHpfiVX73","8rgGU2jaFr"], 100),
 "attack": (["n6iVwAehod","WaU673Xm0t","WqMnQ5EiwI","qSXneBT5CH"], 97),
 "jump":   (["rmFSUCL5Ie","UBVTFOIn6Y","EYDjGUkdB2"], 92),
 "dash": (["Nxjd25gaPm","bK6NKZCrfU","hmne3z5XxA"], 95),
 "hit":  (["TQ0VrB7Kzp","UeLVmiRpkH","tjRB8TPGUS","v0pwJH53K3"], 97),
 "down": (["fVFLuZpQKd","lPcB2ISBl0","mOfvbvdbWb","jN6ULBUqrU"], 77),
}
for g,(ids,target) in NEW.items():
    for i,uid in enumerate(ids,1):
        d=urllib.request.urlopen(m.BASE+uid,timeout=25).read()
        img=Image.open(io.BytesIO(d))
        a=m.remove_bg(np.array(img.convert("RGBA")))
        ys,xs=np.where(a[:,:,3]>10); bh=ys.max()-ys.min()+1
        if g in ("run","walk","attack","jump","dash","hit","down"):
            # 各源图人物绘制大小不一，bbox 归一会让帧间人物忽大忽小；
            # 用不透明面积开方归一（面积∝尺寸²，受姿态影响小），K 调到高度≈95
            K={"run":53.16,"walk":46.5,"attack":54.11,"jump":59.97,
               "dash":66.17,"hit":51.02,"down":56.3}[g]
            m.SCALE=K/np.sqrt((a[:,:,3]>10).sum())
        else:
            m.SCALE=target/bh
        cv=m.refine(img)
        p=os.path.join(m.OUT,f"{g}{i}.png"); cv.save(p)
        aa=np.array(cv)[:,:,3]; yy,xx=np.where(aa>20)
        print(g,i,"src_bh",bh,"out_h",yy.max()-yy.min()+1,"holes",m.audit(cv))
# 更新 manifest 里这两组
man=json.load(open(os.path.join(m.OUT,"manifest.json")))
for g,(ids,_) in NEW.items():
    man[g]=[os.path.join(m.OUT,f"{g}{i}.png") for i in range(1,len(ids)+1)]
json.dump(man,open(os.path.join(m.OUT,"manifest.json"),"w"),ensure_ascii=False,indent=1)
print("done")
