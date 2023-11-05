#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <wiringPi.h>

#include "tft_lib.h"

#include "tjpgd2/tjpgd.h"
#include "tjpgd2/decode_jpeg.h"

#include "pngle/pngle.h"
#include "pngle/decode_png.h"

#include "driver/ili9342.h"
#define DRIVER_NAME "ILI9342"
#define SCREEN_WIDTH	320
#define SCREEN_HEIGHT 240
#define OFFSET_X 0
#define OFFSET_Y 0
#define INIT_FUNCTION(a, b, c, d, e) ili9342_lcdInit(a, b, c, d, e)

time_t elapsedTime(struct timeval startTime, struct timeval endTime) {
	time_t diffsec = difftime(endTime.tv_sec, startTime.tv_sec);
	suseconds_t diffsub = endTime.tv_usec - startTime.tv_usec;
//printf("diffsec=%ld diffsub=%ld\n",diffsec, diffsub);
	if(diffsub < 0) {
		diffsec--;
		diffsub = (endTime.tv_usec+1000000) - startTime.tv_usec;
	}
	uint16_t diffmsec = diffsub / 1000;
	time_t diff = (diffsec * 1000) + diffmsec;
	return diff;
}

time_t PNGTest(TFT_t * dev, char * file, int width, int height) {
	struct timeval startTime, endTime;
	gettimeofday(&startTime, NULL);

	//lcdFillScreen(dev, BLACK);

	int _width = width;
	if (width > 320) _width = 320;
	int _height = height;
	if (height > 320) _height = 320;

	// open PNG file
	FILE* fp = fopen(file, "rb");
	if (fp == NULL) {
		printf("File not found [%s]\n", file);
		return 0;
	}

	char buf[1024];
	size_t remain = 0;
	int len;

	pngle_t *pngle = pngle_new(_width, _height);

	pngle_set_init_callback(pngle, png_init);
	pngle_set_draw_callback(pngle, png_draw);
	pngle_set_done_callback(pngle, png_finish);

	double display_gamma = 2.2;
	pngle_set_display_gamma(pngle, display_gamma);


	while (!feof(fp)) {
		if (remain >= sizeof(buf)) {
			printf("Buffer exceeded\n");
			return 0;
		}

		len = fread(buf + remain, 1, sizeof(buf) - remain, fp);
		if (len <= 0) {
			//printf("EOF\n");
			break;
		}

		int fed = pngle_feed(pngle, buf, remain + len);
		if (fed < 0) {
			printf("ERROR; %s\n", pngle_error(pngle));
			return 0;
		}

		remain = remain + len - fed;
		if (remain > 0) memmove(buf, buf + fed, remain);
	}

	fclose(fp);

	uint16_t pngWidth = width;
	uint16_t offsetX = 0;
	if (width > pngle->imageWidth) {
		pngWidth = pngle->imageWidth;
		offsetX = (width - pngle->imageWidth) / 2;
	}
	//printf("pngWidth=%d offsetX=%d\n", pngWidth, offsetX);

	uint16_t pngHeight = height;
	uint16_t offsetY = 0;
	if (height > pngle->imageHeight) {
		pngHeight = pngle->imageHeight;
		offsetY = (height - pngle->imageHeight) / 2;
	}
	//printf("pngHeight=%d offsetY=%d\n", pngHeight, offsetY);
	uint16_t *colors = (uint16_t*)malloc(sizeof(uint16_t) * pngWidth);

	int ypos = (height-1) - offsetY;
	for(int y = 0; y < pngHeight; y++){
		for(int x = 0;x < pngWidth; x++){
			pixel_png pixel = pngle->pixels[y][x];
			uint16_t color = rgb565_conv(pixel.red, pixel.green, pixel.blue);
			//lcdDrawPixel(dev, x+offsetX, y+offsetY, color);
			lcdDrawPixel(dev, x+offsetX, ypos, color);
		}
		ypos--;
	}


	free(colors);
	pngle_destroy(pngle, _width, _height);

	gettimeofday(&endTime, NULL);
	time_t diff = elapsedTime(startTime, endTime);
	printf("%s elapsed time[ms]=%ld\n",__func__, diff);
	return diff;
}
int main(int argc, char **argv){
	
	if(wiringPiSetup() == -1) {
		printf("wiringPiSetup Fail\n");
		return 1;
	}
	
	char ppath[128];
	
	int i;
	char base[128];
	strcpy(base, argv[0]);
	for(i=strlen(base);i>0;i--) {
		if (base[i-1] == '/') {
			base[i] = 0;
			break;
		}
	}
	strcpy(ppath,base);
	strcat(ppath,"pin.conf");
	TFT_t dev;
	lcdInterface(&dev, ppath);
	lcdReset(&dev);
	INIT_FUNCTION(&dev, SCREEN_WIDTH, SCREEN_HEIGHT, OFFSET_X, OFFSET_Y);

#ifdef INVERT
	lcdInversionOn(&dev);
#endif

	int f,h;
	char file[200];
	for(h=0; h<2;h++){
		for(f=0; f<61;f++){			
			snprintf(file, sizeof(file), "/home/alix/Documents/ALIX/ALIX/DisplayLab/images/neutral/frame%d.png", f);
			PNGTest(&dev, file, SCREEN_WIDTH, SCREEN_HEIGHT);
			sleep(0.01);
		}
	}
	snprintf(file, sizeof(file), "/home/alix/Documents/ALIX/ALIX/DisplayLab/images/neutral/frame0.png", f);
	PNGTest(&dev, file, SCREEN_WIDTH, SCREEN_HEIGHT);
	sleep(0.01);
}
