import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, \
    roc_curve, auc


def evaluate(model, x_test, y_test):
    y_pred = model.predict_classes(x_test)

    print('CLASSIFICATION METRICS\n')
    print(classification_report(
        y_test,
        y_pred,
        target_names=['Negativo', 'Positivo']
    ))

    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='d',
                xticklabels=['Negativo', 'Positivo'],
                yticklabels=['Negativo', 'Positivo']
                )
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.title("CONFUSION MATRIX - Multilayer Perceptron\n", size=16);
    plt.savefig("conf_mat.png")
    plt.show()

    y_pred = model.predict(x_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 8))
    lw = 2
    sns.relplot(x=fpr, y=tpr,  kind='line', size=lw, legend=False)
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='darkorange', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC')
    plt.legend(['ROC curve (area = %0.2f)' % roc_auc], loc="lower right")
    plt.savefig('roc_curve.png')
    plt.show()
