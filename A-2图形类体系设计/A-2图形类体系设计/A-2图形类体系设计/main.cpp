#include<iostream>
#include<cmath>
using namespace std;
//图形抽象类
class Graph 
{
public:
	//面积虚函数
	virtual double AreaCalculation() = 0;
	//周长虚函数
	virtual double CircumetumCalculation() = 0;
	//// 虚析构
	virtual ~Graph() = default;
	//输出函数
	void show()
	{
		cout << "面积：" << AreaCalculation() << "，周长：" << CircumetumCalculation() << endl;
	}
};
//圆形
class Circle :public Graph
{
public:
	double radius;//半径
	Circle(double r)
	{
		radius = r;
	}
	//面积函数
	double AreaCalculation() 
	{
		return 3.1415926 * radius * radius;
	}
	//周长函数
	double CircumetumCalculation()
	{
		return 2 * 3.1415926 * radius;
	}
};
//矩形
class Rectangle :public Graph
{
public:
	double length;//长
	double width;//宽
	Rectangle(double len, double wid)
	{
		length = len;
		width = wid;
	}
	//面积函数
	double AreaCalculation()
	{
		return length * width;
	}
	//周长函数
	double CircumetumCalculation()
	{
		return (width + length) * 2;
	}
};
//三角形
class Triangle :public Graph
{
public:
	//三角形的三条边
	double side1;
	double side2;
	double side3;
	Triangle(double s1, double s2, double s3)
	{
		side1 = s1;
		side2 = s2;
		side3 = s3;
	}
	//检测三角形是否存在
	bool isTriangleValid()
	{
		if ((side1 + side2 > side3) && (side1 + side3 > side2) && (side2 + side3 > side1))
			return true;
		return false;
	}
	//面积函数
	double AreaCalculation()
	{
		double p = (side2 + side1 + side3) / 2;
		return sqrt(p * (p - side1) * (p - side3) * (p - side2));
	}
	//周长函数
	double CircumetumCalculation()
	{
		return side3 + side1 + side2;
	}
};
int main()
{
	cout << "请选择图形：1 圆形，2 矩形，3三角形" << endl;
	int k1;
	cin >> k1;
	if (k1 == 1)
	{
		cout << "请输入圆形的半径" << endl;
		int r;
		cin >> r;
		Graph* s = new Circle(r);
		s->show();
		delete s;
	}
	else if (k1 == 2)
	{
		cout << "请输入长方形的长和宽" << endl;
		int l,w;
		cin >> l>>w;
		Graph* s = new Rectangle(l,w);
		s->show();
		delete s;
	}
	else if (k1 == 3)
	{
		cout << "请输入三角形的三条边" << endl;
		int s1, s2, s3;
		cin >> s1 >> s2 >> s3;
		// 先创建三角形对象
		Triangle tempTri(s1, s2, s3);
		if (tempTri.isTriangleValid())
		{
			Graph* s = new Triangle(s1, s2, s3);
			s->show();
			delete s;
		}
		else
		{
			cout << "不能组成三角形！" << endl;
		}
	}

	system("pause");
	return 0;
}