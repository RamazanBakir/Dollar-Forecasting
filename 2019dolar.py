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

x=x.ravel().reshape(97,1)
y=y.ravel().reshape(97,1)

plt.title("Lineer ve POlinom REgresyon")
plt.xlabel("Gun")
plt.ylabel("Fiyat")
plt.grid()
plt.scatter(x,y)
#plt.show()

#Lineer REgresyon
tahminlineer = LinearRegression()
tahminlineer.fit(x,y) #x ve y eksenine oturtmak için.
tahminlineer.predict(x) #günlere göre fiyatları arıyoruz.Predict tahmin etmesi için.
plt.plot(x,tahminlineer.predict(x),c="red") #çizdirmek için
#print(tahminlineer.predict(x)) #tahminleri yazdırmak için...
#plt.show()

#Polinom Regresyon
#2.dereceden Regresyon için
tahminpolinom = PolynomialFeatures(degree=2)#kaçıncı dereceden polinom olacağını tanımladık.(3.dereceden)
Xyeni = tahminpolinom.fit_transform(x)#x in yeni matrisini oluşturuyor.

polinommodel = LinearRegression() 
polinommodel.fit(Xyeni,y) #Yeni matrisle y değerini eksene oturttuk.
polinommodel.predict(Xyeni) #Yeni matrise göre tahmin ettiriyoruz.
plt.plot(x,polinommodel.predict(Xyeni),c="black") #
#plt.show()
#3.dereceden Regresyon için
tahminpolinom3=PolynomialFeatures(degree=3)
Xyeni=tahminpolinom3.fit_transform(x)

polinommodel3=LinearRegression()
polinommodel3.fit(Xyeni,y)
polinommodel3.predict(Xyeni)
plt.plot(x,polinommodel3.predict(Xyeni),c="green")
#plt.show()
#9.dereceden Regresyon için
tahminpolinom9=PolynomialFeatures(degree=9)
Xyeni=tahminpolinom9.fit_transform(x)

polinommodel9=LinearRegression()
polinommodel9.fit(Xyeni,y)
polinommodel9.predict(Xyeni)
plt.plot(x,polinommodel9.predict(Xyeni),c="gray")
plt.show()
"""
#Hataları görmek için 
verilerimizin tahmin ettiğimiz değerle gerçek değer arasında ki verinin 
karesini alıp toplayıp,tüm değerler için yani bunu yapıp toplayıp,
Hangi modelde daha az hata var yada çok hata var GÖrebiliriz.
"""
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

hatakaresilineer=0
hatakaresipolinom=0

#Polinom regresyonun hatasını görmek için....
for i in range(len(Xyeni)):              #(Gerçek değeriöm - tahmini değerim)**2
    hatakaresipolinom=hatakaresipolinom+(float(y[i]-float(polinommodel.predict(Xyeni)[i])))**2  
for i in range(len(y)):
    hatakaresilineer = hatakaresilineer + (float(y[i])-float(tahminlineer.predict(x)[i]))**2

#Hangi derecenin daha uygun olduğunu bulmak için 
hatakaresipolinom = 0
    
