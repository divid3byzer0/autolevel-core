#include "AutoLevelEngine.h"
#include <cstdio>
#include <chrono>
#include <vector>
#include <cmath>
using namespace autolevel::dsp;
int main(){
    AutoLevelEngine eng; double fs=44100.0; eng.prepare(fs);
    EngineParameters p;                 // shipped defaults
    const size_t block=256; const int blocks=(int)(fs*60.0/block);   // 60 s of audio
    std::vector<float> L(block), R(block);
    double ph=0;
    auto t0=std::chrono::steady_clock::now();
    for(int b=0;b<blocks;++b){
        for(size_t i=0;i<block;++i){ float s=(float)(0.2*std::sin(ph)+0.1*std::sin(ph*3.1)); L[i]=R[i]=s; ph+=2*M_PI*220.0/fs; }
        eng.process(L.data(), R.data(), block, p);
    }
    auto t1=std::chrono::steady_clock::now();
    double sec=std::chrono::duration<double>(t1-t0).count();
    printf("  processed 60.0 s of stereo audio in %.3f s\n", sec);
    printf("  realtime factor: %.1fx   CPU per core: %.2f%%\n", 60.0/sec, 100.0*sec/60.0);
    return 0;
}
