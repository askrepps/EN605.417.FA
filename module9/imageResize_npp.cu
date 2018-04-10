/////////////////////////
// imageResize_npp.cu  // 
// Andrew Krepps       //
// Module 9 Assignment //
// 4/9/2018            //
/////////////////////////

#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <npp.h>

#include <chrono>
#include <sstream>
#include <string>

///////////////////////////////////////////////////////////////////////////////
/// \brief Load an 8-bit grayscale image, resize it, and save the result to an
/// output image file
/// 
/// \param [in] inputFile the path to the input image file
/// \param [in] scaleFactorX the horizontal scaling factor
/// \param [in] scaleFactorY the vertical scaling factor
/// \param [in] interpMode the interpolation mode
/// \param [in] outputFile the path to the desired output image file
/// 
/// \returns total execution time (in ms), not including disk I/O
///////////////////////////////////////////////////////////////////////////////
float resizeImage(
	const std::string& inputFile,
	const float scaleFactorX,
	const float scaleFactorY,
	const NppiInterpolationMode interpMode,
	const std::string& outputFile)
{
	// load 8-bit grayscale image on host
	npp::ImageCPU_8u_C1 srcImage;
	npp::loadImage(inputFile, srcImage);
	
	// use full image as source ROI
	NppiSize srcSize = {(int)srcImage.width(), (int)srcImage.height()};
	NppiRect srcROI = {0, 0, srcSize.width, srcSize.height};
	
	// start clock after image is loaded from disk
	auto start = std::chrono::high_resolution_clock::now();
	
	// copy source image to device
	npp::ImageNPP_8u_C1 d_srcImage(srcImage);
	
	// calculate rescaled destination size/ROI
	const int rescaledX = (int)(srcROI.width*scaleFactorX);
	const int rescaledY = (int)(srcROI.height*scaleFactorY);
	NppiSize dstSize = {rescaledX, rescaledY};
	NppiRect dstROI = {0, 0, dstSize.width, dstSize.height};
	
	// create destination image on device
	npp::ImageNPP_8u_C1 d_dstImage(dstROI.width, dstROI.height);
	
	// resize the image
	NppStatus status = nppiResize_8u_C1R(d_srcImage.data(), d_srcImage.pitch(), srcSize, srcROI,
	                                     d_dstImage.data(), d_dstImage.pitch(), dstSize, dstROI, interpMode);
	
	// only write output image if resize was successful
	float ms = -1.0f;
	if (status == NPP_SUCCESS) {
		// copy result image back to the host
		npp::ImageCPU_8u_C1 dstImage(d_dstImage.size());
		d_dstImage.copyTo(dstImage.data(), dstImage.pitch());
		
		// stop clock after result is transfered back to host
		auto stop = std::chrono::high_resolution_clock::now();	
		std::chrono::duration<float> duration(stop - start);
		ms = duration.count()*1000.0f;
		
		// write result image to the output file
		npp::saveImage(outputFile, dstImage);
	}
	else {
		printf("Error: resize returned NPP status code %d\n", status);
	}
	
	// free device memory
	nppiFree(d_srcImage.data());
	nppiFree(d_dstImage.data());
	
	return ms;
}

int main(int argc, char** argv)
{
	// configure run
	const std::string inputFile = "data/Lena.pgm";
	float scaleFactorX = 1.0f;
	float scaleFactorY = 1.0f;
	NppiInterpolationMode interpMode = NPPI_INTER_NN;
	
	// user can provide no scale arguments for default 1/2 scaling,
	// 1 scale argument for uniform scaling, or 2 scale arguments
	// for independent scaling
	if (argc > 1) {
		scaleFactorX = atof(argv[1]);
	}
	if (argc > 2) {
		scaleFactorY = atoi(argv[2]);
	}
	else {
		scaleFactorY = scaleFactorX;
	}
	
	if (argc > 3) {
		const int interpVal = atoi(argv[3]);
		
		// check for valid interpolation mode
		switch (interpVal) {
			case NPPI_INTER_NN:       // nearest neighbor interpolation (1)
			case NPPI_INTER_LINEAR:   // linear interpolation           (2)
			case NPPI_INTER_CUBIC:    // cubic interpolation            (4)
			case NPPI_INTER_LANCZOS:  // Lanczos filtering              (16)
				interpMode = (NppiInterpolationMode)interpVal;
				break;
			default:
				printf("Error: unsupported interpolation mode value: %d\n", interpVal);
		}
	}
	
	// create output file name from configuration
	std::ostringstream oss;
	oss << inputFile.substr(0, inputFile.rfind('.')) << "_" << scaleFactorX << "_" << scaleFactorY << "_";
	switch (interpMode) {
		case NPPI_INTER_NN:       // nearest neighbor interpolation
			oss << "nn";
			break;
		case NPPI_INTER_LINEAR:   // linear interpolation
			oss << "lin";
			break;
		case NPPI_INTER_CUBIC:    // cubic interpolation
			oss << "cub";
			break;
		case NPPI_INTER_LANCZOS:  // Lanczos filtering
			oss << "lnc";
			break;
	}
	oss << ".pgm";
	
	const std::string outputFile = oss.str();
	
	// run dummy execution to avoid startup performance hit
	resizeImage(inputFile, scaleFactorX, scaleFactorY, interpMode, outputFile);
	
	// time exection of image resize
	float ms = resizeImage(inputFile, scaleFactorX, scaleFactorY, interpMode, outputFile);
	printf("Rescaled image to %s (execution time = %.6f ms)\n", outputFile.c_str(), ms);
	
	return EXIT_SUCCESS;
}
