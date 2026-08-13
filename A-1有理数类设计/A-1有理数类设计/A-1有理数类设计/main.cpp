#include <iostream>
#include <string>
using namespace std;
//求最大公约数
int gcd(int a, int b)
{
    a = abs(a);
    b = abs(b);
    while (b != 0)
    {
        int temp = a % b;
        a = b;
        b = temp;
    }
    return a;
}
//有理数类
class Rational {
    friend Rational operator+(Rational& p1, Rational& p2);
    friend Rational operator-(Rational& p1, Rational& p2);
    friend Rational operator*(Rational& p1, Rational& p2);
    friend Rational operator/(Rational& p1, Rational& p2);
    friend ostream& operator<<(ostream& cout, Rational& p);
    friend Rational simple(Rational& p);
public:
    //分数初始化的化简
    Rational(int Num, int Den)
    {
        int temp = gcd(Num, Den);
        Numerator = Num / temp;
        Denominator = Den / temp;
    }

private:
	int Numerator;//分子
	int Denominator;//分母 
};
//分数的化简
Rational simple(Rational& p)
{
    int temp = gcd(p.Numerator, p.Denominator);
    p.Numerator = p.Numerator / temp;
    p.Denominator = p.Denominator / temp;
    return p;
}
//加法运算符重载
Rational operator+(Rational& p1, Rational& p2)
{
    Rational temp(0,1);
    temp.Numerator = p1.Numerator*p2.Denominator + p2.Numerator*p1.Denominator;
    temp.Denominator = p1.Denominator * p2.Denominator;
    return simple(temp);
}
//减法运算符重载
Rational operator-(Rational& p1, Rational& p2)
{
    Rational temp(0, 1);
    temp.Numerator = p1.Numerator * p2.Denominator - p2.Numerator * p1.Denominator;
    temp.Denominator = p1.Denominator * p2.Denominator;
    return simple(temp);
}
//乘法运算符重载
Rational operator*(Rational& p1, Rational& p2)
{
    Rational temp(0, 1);
    temp.Numerator = p1.Numerator*p2.Numerator;
    temp.Denominator = p1.Denominator * p2.Denominator;
    return simple(temp);
}
//除法运算符重载
Rational operator/(Rational& p1, Rational& p2)
{
    Rational temp(0, 1);
    temp.Numerator = p1.Numerator * p2.Denominator;
    temp.Denominator = p1.Denominator * p2.Numerator;
    return simple(temp);
}
//左移运算符重载
ostream& operator<<(ostream &cout, Rational&p)
{
    if (p.Denominator == 1)
    {
        cout << p.Numerator;
        return cout;
    }
    else 
    {
        cout << p.Numerator << "\\" << p.Denominator;
        return cout;
    }
}
int main()
{
    int a1, a2, b1, b2;
    cout << "输入第一个有理数的分子:";
    cin >> a1;
    cout << "输入第一个有理数的分母:";
    cin >> a2;
    cout << "输入第二个有理数的分子:";
    cin >> b1;
    cout << "输入第二个有理数的分母:";
    cin >> b2;
    Rational a(a1, a2);
    Rational b(b1, b2);
    cout << "选择所需计算1.加 2.减 3.乘 4.除(输入数字）\n";
    int k;
    cin >> k;
    if (k == 1)
    {
        Rational c = a + b;
        cout << c;
    }
    else if(k==2)
    {
        Rational c = a - b;
        cout << c;
    }
    else if(k==3)
    {
        Rational c = a * b;
        cout << c;
    }
    else
    {
        Rational c = a / b;
        cout << c;
    }
	return 0;
}