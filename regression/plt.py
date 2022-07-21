import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from collections import defaultdict
import subprocess
import mw2
#import reg

np.random.seed(0)

with open('Xs2.pkl', 'rb') as f:
	X = pkl.load(f,encoding="latin1")

with open('ys2.pkl', 'rb') as f:
	y = pkl.load(f,encoding="latin1")

with open('corrupted2.pkl','rb') as f:
	corrupted = pkl.load(f,encoding="latin1")

with open('uncorrupted2.pkl','rb') as f:
	uncorrupted = pkl.load(f,encoding="latin1")


smart = True

draw_current_line = True

c = 3
y *= c

N = 1000
eta2 = 0.4
w = [c]
num_delete = 200


for i in corrupted:
	y[i] += np.random.normal(0,3)


corrupted_dict = defaultdict(int)
for i in corrupted:
	corrupted_dict[i] = 1

clean_X = X[uncorrupted,:]
clean_y = y[uncorrupted]
dirty_X = X[corrupted,:]
dirty_y = y[corrupted]


a = np.array([1./N]*N)
top = []
bottom = []

for step in range(4):
	aX = np.matmul(np.diag(a),X)
	if not smart:
		what = np.matmul(np.linalg.pinv(np.matmul(aX.T,X)),np.matmul(aX.T,y))
		resids = np.abs(y - np.matmul(X,what))
		top = list(resids.argsort()[-num_delete:][::-1])
		bottom = list(resids.argsort()[:-num_delete][::-1])
		new_a = np.zeros(N)
		for i in top:
			new_a[i] = 0
		for i in bottom:
			new_a[i] = 1
	else:
		params = 0.000005,2,800,eta2
		what, new_a = mw2.altmin_step(X,y,a,params)
		new_what, new_a = mw2.altmin_step(X,y,new_a,params)
		# what, new_a = reg.altmin_step(X,y,new_a)
		# what, new_a = reg.altmin_step(X,y,new_a)
		# what, new_a = reg.altmin_step(X,y,new_a)
	print("NEW wHAT:", what)
	loss = (1./N)*np.linalg.norm(np.matmul(X[int(eta2*N):,:],what - w),2)**2
	print("LOSS:", loss)
	resids = np.abs(y - np.matmul(X,what))

	fig,ax = plt.subplots()
	plt.axis('off')
	ax.set_xlim([-125,125])
	ax.set_ylim([-440,440])
	for i in range(1000):
		# if step == 3 and corrupted_dict[i] == 1:
		# 	continue
		tup1 = np.array([.255,.541,.702,0.2])
		tup2 = np.array([.965,.573,0,0.2])
		tup1 = np.array([.0,.690,.314,0.4])
		tup2 = np.array([.894,.894,.894,0.2])
		edge_tup1 = np.array([.0,.690,.314,1])
		edge_tup2 = np.array([.459,.459,.459,1])
		color = np.round(a[i] * tup1 + (1 - a[i])*tup2,3)
		edge_color = np.round(a[i] * edge_tup1 + (1 - a[i])*edge_tup2,3)
		plt.scatter(X[i,:],y[i],s=200,marker='.',facecolor=tuple(color),edgecolors=tuple(edge_color),linewidth=.5)

	if draw_current_line:
		# if step == 3:
		plt.plot([-120,120],[-120*w[0],120*w[0]],linewidth=2)
		plt.plot([-120,120],[-120*what[0],120*what[0]],color='black',linewidth=2.5)
		plt.plot([-120,120],[-120*what[0],120*what[0]],color='orange',linewidth=2)
	fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
	if draw_current_line:
		plt.savefig('temp1-step%d.png'%step,pad_inches=0,dpi=300)
		# subprocess.call(['open', 'temp1-step%d.png'%step])
	else:
		plt.savefig('temp1-step%d-noline.png'%step,pad_inches=0,dpi=300)

    # normalize to [0,1]
	if not smart:
		a = new_a.copy()
	else:
		a = new_a.copy() * (1 - eta2) * N
	if smart:
		what = new_what.copy()
# 	fig,ax = plt.subplots()
# 	plt.axis('off')
# 	ax.set_xlim([8,33])
# 	ax.set_ylim([-55,175])	
# 	for i in range(1000):
# 		if not smart:
# 			if topdict[i] == 1:
# 				if step > 0 and (X[i,0] >= 40 or X[i,0] <= 5):
# 					continue
# 				plt.scatter(X[i,:],y[i],s=200,marker='.',facecolor=(.965,.573,0,0.2),edgecolors=(.965,.573,0,1),linewidth=.5)
# 			else:
# 				if step > 0 and (X[i,0] >= 40 or X[i,0] <= 5):
# 					continue
# 				plt.scatter(X[i,:],y[i],s=200,marker='.',facecolor=(.255,.541,.702,0.2),edgecolors=(.255,.541,.702,1),linewidth=.5)
# 		else:
# 			if step > 0 and (X[i,0] >= 40 or X[i,0] <= 5):
# 				continue
# 			tup1 = np.array([.255,.541,.702,0.2])
# 			tup2 = np.array([.965,.573,0,0.2])
# 			color = np.round(a[i] * tup1 + (1 - a[i])*tup2,3)
# 			edge_color = np.round(a[i] * tup1 + (1 - a[i])*tup2,3)
# 			edge_color[-1] = 1.
# 			print a[i], edge_color
# 			plt.scatter(X[i,:],y[i],s=200,marker='.',facecolor=tuple(color),edgecolors=tuple(edge_color),linewidth=.5)
# 	plt.plot([0,45],[0,45*what[0]])
# 	fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
# 	if TEST:
# 		plt.savefig('temp2.png')
# 		subprocess.call(['open', 'temp2.png'])
# 	else:
# 		plt.savefig('bhatia_%d.png'%step,dpi=300)


# 	if not smart:
# 		for i in top:
# 			a[i] = 0
# 	else:
# 		a = new_a.copy()
# 		print a



# fig,ax = plt.subplots()
# plt.axis('off')
# # ax = fig.add_axes([0,0,1,1])

# ax.set_xlim([-125,125])
# ax.set_ylim([-440,440])

# for i in range(len(uncorrupted)):
# 	plt.scatter(clean_X[i,:],clean_y[i],s=200,marker='.',facecolor=(.255,.541,.702,0.2),edgecolors=(.255,.541,.702,1),linewidth=.5)
# fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
# if TEST:
# 	plt.savefig('temp3.png')
# 	subprocess.call(['open', 'temp3.png'])
# else:
# 	plt.savefig('bhatia_clean.png',dpi=300)

# fig,ax = plt.subplots()
# plt.axis('off')
# # ax = fig.add_axes([0,0,1,1])

# ax.set_xlim([-125,125])
# ax.set_ylim([-440,440])

# plt.scatter(clean_X,clean_y,s=200,marker='.',facecolor=(.255,.541,.702,0.2),edgecolors=(.255,.541,.702,1),linewidth=.5)
# plt.scatter(dirty_X,dirty_y,s=200,marker='.',facecolor=(.875,.325,.153,0.2),edgecolors=(.875,.325,.153,1),linewidth=.5)

# fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

# if TEST:
# 	plt.savefig('temp4.png')
# 	subprocess.call(['open', 'temp4.png'])
# else:
# 	plt.savefig('bhatia_dirty.png',dpi=300)

# fig,ax = plt.subplots()
# plt.axis('off')
# # ax = fig.add_axes([0,0,1,1])

# ax.set_xlim([-125,125])
# ax.set_ylim([-440,440])

# plt.scatter(X,y,s=200,marker='.',facecolor=(.255,.541,.702,0.2),edgecolors=(.255,.541,.702,1),linewidth=.5)

# fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)


# if TEST:
# 	plt.savefig('temp5.png')
# 	subprocess.call(['open', 'temp5.png'])
# else:
# 	plt.savefig('bhatia_all.png',dpi=300)
