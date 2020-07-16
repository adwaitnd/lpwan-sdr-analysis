#!/usr/bin/env python3
import numpy as np
import sys
import os
import mixing_tools as mtools
import argparse


def loraSymboltime(bw, sf):
    return (2**sf)/bw


def createOutputFilename(inputfile):
    dirname = os.path.dirname(inputfile)
    resfilename = "mf_" + os.path.basename(inputfile)
    return os.path.join(dirname, resfilename)


def processFile(datafile, outputfile, Ts, bwChirp, fctx, fcrx, sf):
    # load data
    data = mtools.load_IQBinary_int16(datafile)
    dataBB, fcBB, TsBB = mtools.basebandShift(data, Ts, bwSignal=bwChirp,
                        fcSignal=fctx, fcData=fcrx, signalType='ds')

    # construct template and filter
    TChirp = loraSymboltime(bwChirp, sf)
    fcBBChirp = fctx - fcBB
    # print("freq offset of chirp in baseband: ", fcBBChirp)
    template = mtools.chirpTemplate(Ts=TsBB, fc=fcBBChirp, bw=bwChirp, TChirp=TChirp, direction='down')
    # print("size of chirp template: ", template.size)
    result = mtools.matchedFilter(dataBB, template)

    # save results
    np.savez(outputfile, data=result, Ts=TsBB, fc=fcBB)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfile", type=str,
                        help="name of input file")
    parser.add_argument("-o", "--outputfile", type=str,
                        help="name of output")
    parser.add_argument("-r", "--fcrx", type=float,
                        help="receive frequency", default=904.75e6)
    parser.add_argument("-t", "--fctx", type=float,
                        help="transmit frequency of signal", default=905.00e6)
    parser.add_argument("-s", "--sf", type=int,
                        help="spreading factor of signal", default=9)
    parser.add_argument("-b", "--bw", type=float,
                        help="bandwidth of chirp", default=125e3)
    parser.add_argument("-F", "--fs", type=float,
                        help="sampling rate of reception", default=1e6)
    args = parser.parse_args()

    datafile = args.inputfile
    fcrx = args.fcrx
    fctx = args.fctx
    bwChirp = args.bw
    sf = args.sf
    fs = args.fs
    Ts = 1/fs

    # find name of output file
    if args.outputfile:
        resfile = args.outputfile
    else:
        resfile = createOutputFilename(datafile)

    print("processing {}".format(datafile))
    processFile(datafile, resfile, Ts, bwChirp, fctx, fcrx, sf)
    print("output saved to {}".format(resfile))
