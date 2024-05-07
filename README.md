## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
 import pandas as pd
 df=pd.read_csv("/content/Encoding Data.csv")
 df
```
![316296378-eca0e23d-6ea3-4685-ad3e-3a3c688afce4](https://github.com/AdhithiyanK/EXNO-3-DS/assets/121029258/6f301f7a-ff51-4b9a-ba4b-4cdfdb9f3d00)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![316296336-6f0fce59-7852-489d-a76f-db5988a45a3b](https://github.com/AdhithiyanK/EXNO-3-DS/assets/121029258/df0bd9ec-7a6f-4733-be4e-fcbef3e66322)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![316295056-84e9360b-9728-444d-bbe3-f7480e9633f6](https://github.com/AdhithiyanK/EXNO-3-DS/assets/121029258/ecd0d17b-086e-4c2a-8415-0ca0516fd338)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![316295116-addbdb92-ff8a-41f3-af9e-bd97ac6800a2](https://github.com/AdhithiyanK/EXNO-3-DS/assets/121029258/c27bd097-90ba-4996-b651-ee239ca8cbcf)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![316295165-1c7de496-371e-4a21-a189-7ed70ecc2900](https://github.com/AdhithiyanK/EXNO-3-DS/assets/121029258/8312bea9-5b13-4cb9-bd92-7a09511ed871)

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```

![316295256-a8c5038b-2814-4b1d-8f85-c27b292c04d4](https://github.com/AdhithiyanK/EXNO-3-DS/assets/121029258/e81783fd-9150-4722-8aed-53daf9637944)

```
pd.get_dummies(df2,columns=["nom_0"])
```
![316295310-797cb3cf-cc31-4c39-ba12-af1d740bbbed](https://github.com/AdhithiyanK/EXNO-3-DS/assets/121029258/f65e2bee-36fb-4269-ae93-c4470ce5a5c3)

```
pip install --upgrade category_encoders
```
![316295375-eba5c171-3e23-483f-b4ed-6f4c47b5e89b](https://github.com/AdhithiyanK/EXNO-3-DS/assets/121029258/5869e6d6-22e5-48a0-a340-7958f2e4869e)

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
fb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![316295450-9b983cb5-c712-49d8-bcd8-817ca7f56947](https://github.com/AdhithiyanK/EXNO-3-DS/assets/121029258/18601b2d-cf5d-46cd-b432-417efa72863a)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![316295534-0d2e7e51-9cb4-4530-a56d-39d1435d0249](https://github.com/AdhithiyanK/EXNO-3-DS/assets/121029258/40ca07bc-b724-4d66-957c-a07cf727d576)

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![316295585-f71a4dd5-6389-4fb9-a209-81064b1d878f](https://github.com/AdhithiyanK/EXNO-3-DS/assets/121029258/c7adeede-a587-4e38-90ae-67eeb45ad360)

```
df.skew()
```
![316295623-3e30db2a-11cd-4980-a53d-87e495cdba2d](https://github.com/AdhithiyanK/EXNO-3-DS/assets/121029258/3238f0e4-a547-4703-bc38-6f3a3c8163ab)

```
np.log(df["Highly Positive Skew"])
```
![316295660-ea34c360-81a5-4760-bc9e-2c6a9d07be6c](https://github.com/AdhithiyanK/EXNO-3-DS/assets/121029258/cd1b02b1-ff63-45d2-8b38-560910bfecc3)

```
np.reciprocal(df["Moderate Positive Skew"])
```
![316295703-790eadf6-e770-4874-943c-ddfb7bca5113](https://github.com/AdhithiyanK/EXNO-3-DS/assets/121029258/edd5541b-1856-4795-9fe2-ce8039814c90)

```
np.sqrt(df["Highly Positive Skew"])
```
![316295749-7e2ab48c-7321-477e-81e4-ba3d6b132bf5](https://github.com/AdhithiyanK/EXNO-3-DS/assets/121029258/fdff6ce5-b38b-4d52-9947-61eedf3be8bf)

```
np.square(df["Highly Positive Skew"])
```
![316295806-bcda767c-4193-4548-89ae-8ba49be8a323](https://github.com/AdhithiyanK/EXNO-3-DS/assets/121029258/75e276dc-caf4-423f-b99d-41d650fe5379)

```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"]) df
```
![316295883-7632047a-430d-4b18-8bab-abc86a475a5a](https://github.com/AdhithiyanK/EXNO-3-DS/assets/121029258/087b28e5-0803-425c-82ed-efbd2b0fccfd)

```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```
![316295945-90edf1a6-0480-49e3-9fe6-7ac276d2acf3](https://github.com/AdhithiyanK/EXNO-3-DS/assets/121029258/9e9f06c2-c694-4352-bf67-2962ac6b3e2a)

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![316295988-af78ba87-bd09-43e8-a612-406cad9d7686](https://github.com/AdhithiyanK/EXNO-3-DS/assets/121029258/413dba80-9db1-4baa-928d-cb402eeaf379)

```
import matplotlib.pyplot as plt import seaborn as sns import statsmodels.api as sm import scipy.stats as stats
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![316296082-b72a5319-b492-4c91-981e-15bf8d38c539](https://github.com/AdhithiyanK/EXNO-3-DS/assets/121029258/26e25307-c77b-4460-9d28-1bbd7e6bffcf)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
```
![316296150-9c011486-2fd5-4752-b434-702ecc5bebdc](https://github.com/AdhithiyanK/EXNO-3-DS/assets/121029258/55e362a7-75e6-4ea4-bede-780188301948)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![316296228-880ace95-7dd1-4732-941d-e007439a6fc5](https://github.com/AdhithiyanK/EXNO-3-DS/assets/121029258/3e5e422b-599e-4bf0-8829-f8abdf99e7d6)




# RESULT:
   Hence performing Feature Encoding and Transformation process is Successful.

       
