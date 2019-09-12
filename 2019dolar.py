#! /usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression #lineer regresyon kütüphanesi
from sklearn.preprocessing import PolynomialFeatures #polinom regresyon kütüphanesi

veri = pd.read_csv("2019dolaralis.csv")

x = veri["Gun"]
y = veri["Fiyat"]


x=x.ravel().reshape(len(veri),1)
y=y.ravel().reshape(len(veri),1)

plt.title("Lineer ve POlinom REgresyon")
plt.xlabel("Gun")
plt.ylabel("Fiyat")
plt.grid()
plt.scatter(x,y)
#plt.show()

#Lineer Regresyon
tahminlineer=LinearRegression()
tahminlineer.fit(x,y) #x ve y eksenine oturtmak için.
tahminlineer.predict(x) #günlere göre fiyatları arıyoruz.Predict tahmin etmesi için.
plt.plot(x,tahminlineer.predict(x),c="red") #çizdirmek için
#print(tahminlineer.predict(x)) #tahminleri yazdırmak için...
#plt.show()

#Polinom Regresyon
a=[2,3,9]
b=["black","green","gray"]
i=0
for i in range(len(a)):
    tahminpolinom = PolynomialFeatures(degree=a[i])#kaçıncı dereceden polinom olacağını tanımladık.
    Xyeni = tahminpolinom.fit_transform(x)#x in yeni matrisini oluşturuyor.

    polinommodel = LinearRegression() 
    polinommodel.fit(Xyeni,y) #Yeni matrisle y değerini eksene oturttuk.
    polinommodel.predict(Xyeni) #Yeni matrise göre tahmin ettiriyoruz.
    plt.plot(x,polinommodel.predict(Xyeni),c=b[i]) #
    i=i+1
plt.show()
#Hangi derecenin daha iyi olduğunu bulmak için.
a=0
hatakaresipolinom=0
for a in range(50):

    tahminpolinom = PolynomialFeatures(degree=a+1)
    Xyeni = tahminpolinom.fit_transform(x)

    polinommodel = LinearRegression()
    polinommodel.fit(Xyeni,y)
    polinommodel.predict(Xyeni)
    for i in range(len(Xyeni)):
        hatakaresipolinom = hatakaresipolinom + (float(y[i])-float(polinommodel.predict(Xyeni)[i]))**2
    print("----------")
    print(a+1,"inci dereceden fonksiyonda hata,", hatakaresipolinom)
    print("----------")


    hatakaresipolinom = 0

"""
#Hataları görmek için 
verilerimizin tahmin ettiğimiz değerle gerçek değer arasında ki verinin 
karesini alıp toplayıp,tüm değerler için yani bunu yapıp toplayıp,
Hangi modelde daha az hata var yada çok hata var GÖrebiliriz.
"""
hatakaresilineer=0
hatakaresipolinom=0
#Polinom regresyonun hatasını görmek için....
for i in range(len(Xyeni)):              #(Gerçek değerim - tahmini değerim)**2
    hatakaresipolinom=hatakaresipolinom+(float(y[i]-float(polinommodel.predict(Xyeni)[i])))**2  
for i in range(len(y)):
    hatakaresilineer = hatakaresilineer + (float(y[i])-float(tahminlineer.predict(x)[i]))**2
