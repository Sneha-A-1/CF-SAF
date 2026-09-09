import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pickle, os
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
 'axes.labelsize':14,'xtick.labelsize':11,'ytick.labelsize':11,
 'legend.fontsize':10.5,'axes.grid':True,'grid.linestyle':'--',
 'grid.alpha':0.45,'grid.color':'#cccccc'})
import paths
OUT=paths.FIGURES
M=pickle.load(open(os.path.join(paths.FIGDATA,'main.pkl'),'rb'))
S=pickle.load(open(os.path.join(paths.FIGDATA,'sweeps.pkl'),'rb'))
ORDER=['Cloud-Only','Naive Fog','Threshold-Fog','FuzzyOff',
       'HEFT-Fog','FuzzyDRL-Fog','CF-SAF']
COL={'Cloud-Only':'#E53935','Naive Fog':'#FF9800','Threshold-Fog':'#FFC107',
 'FuzzyOff':'#9C27B0','HEFT-Fog':'#2196F3','FuzzyDRL-Fog':'#E91E63',
 'CF-SAF':'#4CAF50'}
lat={k:M[k]['metrics']['latency_ms'] for k in ORDER}

# ---- Fig 2: mean latency (log scale, Naive Fog is 40x others) ----
f,a=plt.subplots(figsize=(10,5.2))
b=a.bar(range(7),[lat[k] for k in ORDER],color=[COL[k] for k in ORDER],
        alpha=.88,edgecolor='black',linewidth=.6)
a.set_yscale('log')
for i,k in enumerate(ORDER):
    a.text(i,lat[k]*1.12,f'{lat[k]:.1f}',ha='center',fontsize=10.5,fontweight='bold')
a.set_ylabel(r'Mean Task Latency $\bar{L}$ (ms, log scale)')
a.set_xticks(range(7)); a.set_xticklabels(ORDER,rotation=22,ha='right')
a.set_ylim(100,30000)
plt.tight_layout(); plt.savefig(f'{OUT}/Figure2.png',dpi=200,bbox_inches='tight',facecolor='white')
plt.close()

# ---- Fig 3: normalised multi-metric ----
f,a=plt.subplots(figsize=(11,5.2))
mets=['latency_ms','exec_ms','energy_j']
labs=['Norm. Latency','Norm. Exec. Time','Norm. Energy']
cols=['#E74C3C','#3498DB','#2ECC71']
x=np.arange(7); w=.26
for j,(mt,lb,c) in enumerate(zip(mets,labs,cols)):
    base=M['Cloud-Only']['metrics'][mt]
    v=[M[k]['metrics'][mt]/base for k in ORDER]
    a.bar(x+(j-1)*w,v,w,label=lb,color=c,alpha=.88,edgecolor='black',linewidth=.5)
a.axhline(1.0,color='#444',ls=':',lw=1.3)
a.set_yscale('log')
a.set_ylabel('Normalised Value (Cloud-Only = 1, log)')
a.set_xticks(x); a.set_xticklabels(ORDER,rotation=22,ha='right')
a.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.22),frameon=False)
plt.tight_layout(); plt.savefig(f'{OUT}/Figure_3.png',dpi=200,bbox_inches='tight',facecolor='white')
plt.close()

# ---- Fig 4: latency CDF ----
f,a=plt.subplots(figsize=(10,5.4))
for k in ORDER:
    d=np.sort(M[k]['lat']); y=np.arange(1,len(d)+1)/len(d)
    lw=3.0 if k=='CF-SAF' else 1.8
    a.plot(d,y,label=k,color=COL[k],lw=lw)
p50={k:np.median(M[k]['lat']) for k in ['CF-SAF','Cloud-Only']}
for k,c in [('CF-SAF','#4CAF50'),('Cloud-Only','#E53935')]:
    a.axvline(p50[k],color=c,ls='--',lw=1.3,alpha=.8)
    a.text(p50[k]*1.05,.45,f'$P_{{50}}$={p50[k]:.0f}',color=c,fontsize=9.5,rotation=90)
a.set_xscale('log'); a.set_xlim(50,40000)
a.set_xlabel('Task Latency $L$ (ms, log scale)')
a.set_ylabel('Cumulative Probability $F(L)$')
a.legend(loc='lower right',framealpha=.93)
plt.tight_layout(); plt.savefig(f'{OUT}/Figure4.png',dpi=200,bbox_inches='tight',facecolor='white')
plt.close()

# ---- Fig 5: per-node utilisation ----
f,a=plt.subplots(figsize=(12.5,5.6))
show=['Naive Fog','FuzzyOff','HEFT-Fog','FuzzyDRL-Fog','CF-SAF']
x=np.arange(10); w=.16
for n,k in enumerate(show):
    u=M[k]['util']
    a.bar(x+(n-2)*w,u,w,label=k,color=COL[k],alpha=.88,edgecolor='white',linewidth=.5)
    a.axhline(np.mean(u),color=COL[k],ls='--',lw=1.2,alpha=.7)
    a.text(9.62,np.mean(u)+.6,f'{np.mean(u):.0f}',color=COL[k],fontsize=9,fontweight='bold')
a.axhline(80,color='#333',ls=':',lw=1.6)
a.text(-.45,81.5,r'$\Lambda_{\mathrm{thresh}}=80\%$',fontsize=10,color='#333',fontweight='bold')
a.set_xlabel('Fog Node'); a.set_ylabel(r'Resource Utilisation $\eta_i$ (%)')
a.set_xticks(x); a.set_xticklabels([f'Node {i}' for i in range(1,11)])
a.set_ylim(0,105); a.set_xlim(-.6,9.95)
a.legend(ncol=5,loc='upper center',bbox_to_anchor=(.5,-.14),frameon=False)
plt.tight_layout(); plt.savefig(f'{OUT}/Figure5.png',dpi=200,bbox_inches='tight',facecolor='white')
plt.close()

# ---- Fig 6: energy + throughput ----
f,a=plt.subplots(figsize=(11,5.2))
en=[M[k]['metrics']['energy_j'] for k in ORDER]
tp=[M[k]['metrics']['throughput'] for k in ORDER]
a.bar(range(7),en,.55,color='#E8836F',alpha=.85,edgecolor='black',
      linewidth=.6,label='Energy (J/task)')
a.set_yscale('log'); a.set_ylabel(r'Energy $\mathcal{E}$ (J/task, log)')
for i,v in enumerate(en): a.text(i,v*1.1,f'{v:.2f}',ha='center',fontsize=9.5)
a2=a.twinx(); a2.plot(range(7),tp,'o-',color='#2E7D32',lw=2.2,ms=9,
        mfc='white',mew=2,label='Throughput (tasks/s)')
a2.set_ylabel(r'Throughput $\Theta$ (tasks/s)',color='#2E7D32')
a2.tick_params(axis='y',colors='#2E7D32'); a2.grid(False)
a.set_xticks(range(7)); a.set_xticklabels(ORDER,rotation=22,ha='right')
h1,l1=a.get_legend_handles_labels(); h2,l2=a2.get_legend_handles_labels()
a.legend(h1+h2,l1+l2,loc='upper right',framealpha=.93)
plt.tight_layout(); plt.savefig(f'{OUT}/Figure6.png',dpi=200,bbox_inches='tight',facecolor='white')
plt.close()

# ---- Fig 7: scalability ----
f,a=plt.subplots(figsize=(10,5.4))
mk={'Cloud-Only':'s','Naive Fog':'^','Threshold-Fog':'D','FuzzyOff':'v',
 'HEFT-Fog':'o','FuzzyDRL-Fog':'p','CF-SAF':'*'}
for k in [x for x in ORDER if x in S['scal']]:
    lw=2.8 if k=='CF-SAF' else 1.8; ms=13 if k=='CF-SAF' else 7
    a.plot(S['rates'],S['scal'][k],marker=mk[k],color=COL[k],lw=lw,ms=ms,label=k)
a.axvline(270,color='#666',ls=':',lw=1.6)
a.text(272,a.get_ylim()[1]*0.35,'stability\nlimit',fontsize=9,color='#444')
a.set_yscale('log')
a.set_xlabel('Task Arrival Rate (tasks/second)')
a.set_ylabel(r'Mean Task Latency $\bar{L}$ (ms, log)')
a.legend(loc='upper left',framealpha=.93,ncol=2)
plt.tight_layout(); plt.savefig(f'{OUT}/Figure7.png',dpi=200,bbox_inches='tight',facecolor='white')
plt.close()

# ---- Fig 8: theta sensitivity ----
f,a=plt.subplots(figsize=(10,5.0))
a.plot(S['thetas'],S['lat'],'o-',color='#4CAF50',lw=2.6,ms=8,
       label=r'Mean Latency $\bar{L}$')
a.set_xlabel(r'IDCRA Routing Threshold $\theta^*$')
a.set_ylabel(r'Mean Task Latency $\bar{L}$ (ms)',color='#2E7D32')
a.tick_params(axis='y',colors='#2E7D32')
a2=a.twinx(); a2.plot(S['thetas'],S['f1'],'s--',color='#2196F3',lw=2.2,ms=7,
        label=r'$F_1$ Score (urgent class)')
a2.set_ylabel(r'IDCRA $F_1$ Score',color='#1565C0')
a2.tick_params(axis='y',colors='#1565C0'); a2.grid(False)
a.axvline(0.22,color='#666',ls=':',lw=1.6)
a.text(0.23,max(S['lat'])*.75,r'$\theta^*=0.22$',fontsize=10,color='#444')
h1,l1=a.get_legend_handles_labels(); h2,l2=a2.get_legend_handles_labels()
a.legend(h1+h2,l1+l2,loc='upper right',framealpha=.93)
plt.tight_layout(); plt.savefig(f'{OUT}/Figure8.png',dpi=200,bbox_inches='tight',facecolor='white')
plt.close()

# ---- Fig 9: stability / operating point ----
D=S['drift']
f,a=plt.subplots(figsize=(9,4.8))
a.plot(D['rates'],D['n4k'],'o-',color='#2196F3',lw=2.2,ms=9,label='4,000 tasks/run')
a.plot(D['rates'],D['n12k'],'s-',color='#E53935',lw=2.2,ms=9,label='12,000 tasks/run')
a.axvline(270,color='#4CAF50',ls='--',lw=2)
a.text(270.4,300,'highest stable rate\n270 tasks/s',color='#2E7D32',fontsize=10,fontweight='bold')
a.set_xlabel('Task Arrival Rate (tasks/second)')
a.set_ylabel(r'Mean Task Latency $\bar{L}$ (ms)')
a.set_title('Steady-state check: identical latency at both run lengths\nindicates a stable operating point',fontsize=11)
a.legend(loc='upper left',framealpha=.93)
plt.tight_layout(); plt.savefig(f'{OUT}/Figure9.png',dpi=200,bbox_inches='tight',facecolor='white')
plt.close()
print("all figures written")
for k in ORDER: print(f"  {k:<15} lat={lat[k]:8.2f}  util={np.mean(M[k]['util']):5.1f}")
