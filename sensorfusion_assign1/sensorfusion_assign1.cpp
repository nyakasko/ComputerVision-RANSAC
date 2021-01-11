#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
#include "sensorfusion_assign1.h"
#include <time.h>
#include <vector>

using namespace std;
using namespace cv;

void FitLineRANSAC(
	vector<Point> points_,
	double threshold_,
	int maximum_iteration_number_,
	Mat image_)
{
	srand(time(NULL));
	int iterationNumber = 0;

	vector<int> inliers;
	inliers.reserve(points_.size());

	constexpr int kSampleSize = 2;
	std::vector<int> sample(kSampleSize);
	bool shouldDraw = image_.data != nullptr;
	while (iterationNumber++ < maximum_iteration_number_)
	{
		// 1. Select a minimal sample, i.e., in this case, 2 random points.
		for (size_t sampleIdx = 0; sampleIdx < kSampleSize; ++sampleIdx)
		{
			do
			{
				sample[sampleIdx] =
					round((points_.size() - 1) * static_cast<double>(rand()) / static_cast<double>(RAND_MAX));
				if (sampleIdx == 0)
					break;
				if (sampleIdx == 1 &&
					sample[0] != sample[1])
					break;
			} while (true);
		}
		// 2. Fit a line to the points.
		const Point2d& p1 = points_[sample[0]]; // First point selected
		const Point2d& p2 = points_[sample[1]]; // Second point select		
		Point2d v = p2 - p1; // Direction of the line
		v = v / cv::norm(v); // Division by the length of the vector to make it unit length
		Point2d n; // Normal of the line (perpendicular to the line)
		// Rotate v by 90° to get n
		n.x = -v.y;
		n.y = v.x;
		// distance(line, point) = | a * x + b * y + c | / sqrt(a * a + b * b)
		// if ||n||_2 = 1 (unit length) then sqrt(a * a + b * b) = 1 and I don't have to do the division that is in the previous line
		long double a = n.x;
		long double b = n.y;
		long double c = -(a * p1.x + b * p1.y);

		// 3. Count the number of inliers, i.e., the points closer than the threshold.
		inliers.clear();
		for (size_t pointIdx = 0; pointIdx < points_.size(); ++pointIdx)
		{
			const Point2d& point = points_[pointIdx];
			const long double distance =
				abs(a * point.x + b * point.y + c);

			if (distance < threshold_)
			{
				inliers.emplace_back(pointIdx);
			}
		}

		if (inliers.size() > 150 && shouldDraw)
		{
			for (size_t pointIdx = inliers.size()-1; pointIdx > 0; --pointIdx)
			{
				points_.erase(points_.begin() + inliers[pointIdx]);
			}
			inliers.clear();
			inliers.resize(0);
			cv::line(image_,
				p1,
				p2,
				cv::Scalar(0, 0, 255),
				1);
		}
	}
}

void DrawPoints(vector<Point>& points, Mat image)
{
	for (int i = 0; i < points.size(); ++i)
	{
		circle(image, points[i], 1, Scalar(255, 255, 255));
	}
}

int main()
{
	Mat input = imread("right.jpg", 1);

	if (input.data == nullptr) {
		cerr << "Failed to load image" << endl;
	}
	Mat gray;
	Mat edge_gaus;
	int kernel_size = 5;
	Mat blur_gray;
	cvtColor(input, gray, CV_BGR2GRAY);
	GaussianBlur(gray, blur_gray, Size(3, 3), 1.0);
	Canny(blur_gray, edge_gaus, 50, 150, 3, true);
	//imshow("Gaus", edge_gaus);
	Mat image = Mat::zeros(edge_gaus.rows, edge_gaus.cols, CV_8UC3);
	vector<Point> points;

	findNonZero(edge_gaus, points);
	//DrawPoints(points, input);
	FitLineRANSAC(points, 0.5, 100000, input);

	imshow("Edge detected image", edge_gaus);
	imshow("Output", input);
	waitKey(0);

	return 0;
}
 