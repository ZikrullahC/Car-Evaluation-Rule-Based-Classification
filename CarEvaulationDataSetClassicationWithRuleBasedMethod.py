import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Veri setini yükleyin
data = pd.read_csv("car_dataset.csv")

# Veri setinin başlıklarını kontrol edin
print(data.head())

def rule_based_classifier(row):
    if (row['buying'] == 'vhigh' and (row['maint'] == 'vhigh' or row['maint'] == 'high')):
        return 'unacc'
    elif(row['buying'] == 'high' and row['maint'] == 'vhigh'):
        return 'unacc'
    elif(row['safety'] == 'low' or row['persons'] == '2'):
        return 'unacc'
    # Bu kural vhigh,high veya high,high durumlarına göre çalışır.
    elif (row['buying'] == 'high' and row['maint'] == 'high' or row['maint'] == 'med'):
        if(row['lug_boot'] == 'big' or row['lug_boot'] == 'med'):
            return 'acc'
        elif(row['safety'] == 'high'):
            return 'acc'
        else :
            return 'unacc'
    elif(row['buying'] == 'med'):
        if(row['maint'] == 'high'):
            if(row['lug_boot'] == 'big' or row['lug_boot'] == 'med'):
                return 'acc'
            elif(row['safety'] == 'high'):
                return 'acc'
            else :
                return 'unacc'   
        elif(row['maint'] == 'med'):
            # maint == med veya low olur. lug_boot == small degilse ve safety == high ise vgood olur
            if(row['safety'] == 'high' and row['lug_boot'] != 'small'):
                return 'vgood'
            # Çünkü safety == low değilse her durumda acc olur.
            else:
                return 'acc'
        # maint == low
        else:
            if(row['lug_boot'] != 'big' and row['safety'] == 'high'):
                return 'good'
            elif(row['lug_boot'] == 'big' and row['safety'] == 'high'):
                return 'vgood'
            elif(row['lug_boot'] == 'big' and row['safety'] == 'med'):
                return 'good'
            else:
                return 'acc'
            
    elif(row['buying'] == 'low'):
        if(row['maint'] == 'vhigh'):
            if(row['lug_boot'] == 'small' or row['lug_boot'] == 'med'):
                if(row['safety'] == 'high'):
                    return 'acc'
                else:
                    return 'unacc'
            # Eger safety == low degilse ve lug_boot == big ise --> acc
            else:
                return 'acc'
        elif(row['maint'] == 'high'):
            if(row['lug_boot'] == 'big' and row['safety'] == 'high'):
                return 'vgood'
            elif(row['lug_boot'] == 'small'):
                return 'acc'
            else:
                return 'unacc'
        elif(row['maint'] == 'med'):
            if(row['lug_boot'] == 'small' or row['lug_boot'] == 'med'):
                if(row['safety'] == 'high'):
                    return 'good'
                else:
                    return 'acc'
            # lug_boot == big    
            else:
                if(row['safety'] == 'high'):
                    return 'vgood'
                else:
                    return 'good'
        elif(row['maint'] == 'low'):
            if(row['lug_boot'] == 'small' or row['lug_boot'] == 'med'):
                if(row['safety'] == 'med'):
                    return 'acc'
                else:
                    return 'good'
            elif(row['lug_boot'] == 'big'):
                if(row['safety'] == 'med'):
                    return 'good'
                else:
                    return 'vgood'
        # else:
        #     return 'unacc'
    else:
        return 'unknown'  # Belirli bir koşulu karşılamayan durumlar için

# Rule-based sınıflandırmayı uygulayın
data['predicted_class'] = data.apply(rule_based_classifier, axis=1)

# Gerçek sınıflarla tahmin edilen sınıfları alın
y_true = data['ClassValues']
y_pred = data['predicted_class']

# Confusion matrix oluşturun
cm = confusion_matrix(y_true, y_pred)

# Performans metriklerini hesaplayın
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print("Confusion Matrix:")
print(cm)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Metrikleri görselleştirin
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(8, 6))
plt.bar(metrics, values, color='skyblue')
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Performance Metrics of Rule-Based Classifier')
plt.ylim(0, 1)  # Değer aralığını belirleyin (0 ile 1 arası)
plt.show()