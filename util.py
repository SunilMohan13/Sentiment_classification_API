# import modules
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def plot_roc(model, x_columns, y_true, size_x=12, size_y=12):
	y_pred =  model.predict_proba(x_columns)
	
	fpr, tpr, threshold = roc_cureve(y_true, y_pred[:,1])
	area_under_curve = auc(fpr, tpr)
	
	fig, ax = plt.suplots(figsize=(size_x, size_y))
	model_name = str(type(model)).split('.')[-1].strip(">\'")
	plt.title(f'{model_name} ROC')
	ax.plot(fpr, tpr, 'k', label='AUC = %0.3f' % area_under_curve)
	
	ax.legend(loc='lower right')
	ax.plot([0, 1], [0, 1], 'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.show()