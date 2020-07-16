import numpy as np
from scipy.constants import c as c0  # speed of light
import sqlite3

### general helper functions ###

# skip comments and get to first line of data
def skipComments(fp):
    # note: tested against a file which is only made of comments
    lineStart = fp.tell()
    while True:
        if fp.readline().startswith('#'):
            lineStart = fp.tell()
        else:
            # first line that is not a comment has been read already
            fp.seek(lineStart)
            break

def parseNoOfTotalRX(line):
    return int(line)

def parseNoOfPaths(line):
    rxID, nPaths = list(int(s) for s in line.split())
    return rxID, nPaths


# takes a line of p2m file text and returns a tuple of (toa, complex power (mW))
def parseCIREntry(line):
    array = line.split()
    # pathNo = int(array[0])
    phase = float(array[1])  # degrees
    toa = float(array[2])  # sec
    Pdbm = float(array[3])  # dbm
    P = np.power(10, Pdbm/10.0)*np.exp(1j*np.radians(phase))  # complex power in mW
    return toa, P

def parseDelaySpreadEntry(line):
    """takes a line from the p2m file an outputs a tuple of (rxid, position, distance, delay spread)

    Arguments:
        line {[string]} -- [line from p2m file]

    Returns:
        [tuple] -- [( rxid (int), pos (np.array(3,)), dist (float), delaySpread (float) )]
    """    
    array = line.split()
    rxid = int(array[0])
    x = float(array[1])
    y = float(array[2])
    z = float(array[3])
    pos = np.array([x,y,z])
    dist = float(array[4])
    ds = float(array[5])
    return rxid, pos, dist, ds

### functions for external use

# load the impulse response of a particular receiver from file
def loadEntryCIR(file, rxID):
    with open(file, 'r') as fp:

        skipComments(fp)

        # get number of receivers (1st line of data)
        nRX = int(fp.readline())  # note: int() takes care of stray whitespace at the beginning

        # print("{:d} receivers in file".format(nRX))
        # go to entry for the nth receiver
        if rxID > nRX:
            raise IndexError("rxID out of range")

        # skip everything upto required receiver ID
        for _ in range(rxID-1):
            line = fp.readline()
            _, nrl = list(int(s) for s in line.split())
            for _ in range(nrl):
                _ = fp.readline()

        # we should be at requested receiver, but double check anyway
        rID, nrl = parseNoOfPaths(fp.readline())
        if rxID != rID:
            raise IndexError("rxID not found at expected location")
        pathList = []
        for _ in range(nrl):
            pathList.append(parseCIREntry(fp.readline()))

        return pathList

    
# load the impulse response of all receivers as a dict
def loadAllCIR(file, rxset=None, txset=None, txid=None):
    with open(file, 'r') as fp:

        skipComments(fp)

        # get number of receivers (1st line of data)
        nRX = int(fp.readline())  # note: int() takes care of stray whitespace at the beginning

        # print("{:d} receivers in file".format(nRX))
        cirDict = {}
        for _ in range(nRX):
            rxID, nrl = parseNoOfPaths(fp.readline())
            pathList = []
            for _ in range(nrl):
                pathList.append(parseCIREntry(fp.readline()))
            cirDict[((txset, txid),(rxset, rxID))] = pathList
        return cirDict

def sqlfilename(project, studyarea, projectdir=None):
    """output the name and location of an sqlite file for a particular study

    Args:
        project ([type]): [description]
        studyarea ([type]): [description]
        projectdir ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """    
    filename = "{:s}.{:s}.sqlite".format(project, studyarea)
    if projectdir:
        return "{:s}/{:s}/{:s}".format(projectdir, studyarea, filename)
    else:
        return "{:s}/{:s}".format(studyarea, filename)


def outputfilename(project, studyarea, output, txset, txn, rxset, projectdir=None):
    """output the name of a p2m file for a particular output in a study

    Args:
        project (string): [description]
        studyarea (string): [description]
        output (string): [description]
        txset (integer): [description]
        txn (integer): [description]
        rxset (integer): [description]
        projectdir (string, optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """    
    filename = "{:s}.{:s}.t{:03d}_{:02d}.r{:03d}.p2m".format(project, output, txn, txset, rxset)
    if projectdir:
        return "{:s}/{:s}/{:s}".format(projectdir, studyarea, filename)
    else:
        return "{:s}/{:s}".format(studyarea, filename)


# a function to parse SQL entries for device positions
def parsePos(sqlfile, devType, setid=None):
    """parse device positions from SQL entries

    Arguments:
        sqlfile {string} -- [name of SQLite file]
        devType {string} -- [select type of device: "tx", "rx", "txset" or "rxset"]

    Raises:
        ValueError: [invalid type of device]

    Returns:
        [dict] -- [returns a dictionary of locations {(setid, devid): np.array([x,y,z])}]
    """    
    # fetch all entries
    con = sqlite3.connect(sqlfile)
    cur = con.cursor()
    if devType == "tx":
        # grab all TX
        cur.execute("SELECT tx_id, tx_set_id, x, y, z FROM tx")
    elif devType == "rx":
        # grab all RX
        cur.execute("SELECT rx_id, rx_set_id, x, y, z FROM rx")
    elif devType == "txset":
        # grab certain TX sets only
        cur.execute("SELECT tx_id, tx_set_id, x, y, z FROM tx WHERE tx_set_id=?", str(setid))
    elif devType == "rxset":
        # grab certain RX sets only
        cur.execute("SELECT rx_id, rx_set_id, x, y, z FROM rx WHERE rx_set_id=?", str(setid))
    else:
        raise ValueError("unknown devType")
    entries = cur.fetchall()
    con.close()
    
    # parse all sql entries
    locDict = {}  # final dict that hols location outputs
    offsetDict = {}  # intermediate dict that holds the offsets 
    for entry in entries:
        tableid, setid, x, y, z = entry
        if setid not in offsetDict:
            # we have never seen this ID before. Add an entry for it
            offsetDict[setid] = tableid
        devid = tableid + 1 - offsetDict[setid]
        loc = np.array([x,y,z])
        locDict[(setid, devid)] = loc
    return locDict

# get distances between certain transmitter and receiver sets
def computeDistancesLOS(txPosSet, rxPosSet):
    distances = {}
    # yes, this could be optimized rather than using nested loops
    for txid, txpos in txPosSet.items():
        for rxid, rxpos in rxPosSet.items():
            distances[(txid, rxid)] = np.linalg.norm(txpos - rxpos)
    return distances


# get LOS time of arrival
def computeToaLOS(distDict):
    toa = {}
    for ids, dist in distDict.items():
        toa[ids] = dist/c0
    return toa


def loadDelaySpread(file, rxset=None, txset=None, txid=None):
    with open(file, 'r') as fp:

        skipComments(fp)

        # get number of receivers (1st line of data)
        posDict = {}
        distDict = {}
        delaySpreadDict = {}
        for line in fp:
            rxid, pos, dist, ds = parseDelaySpreadEntry(line)
            entryid = ((txset, txid), (rxset, rxid))  # entries for distance and delay spread are identified by both tx and rx involved
            posDict[(rxset, rxid)] = pos  # position dictionary entries are only identified by the rxset and rxid
            distDict[entryid] = dist
            delaySpreadDict[entryid] = ds
        
        return delaySpreadDict, posDict, distDict


def resampleCIR(cirList, Ts, tEnd, tOff=0):
    tsamples = np.arange(tOff, tEnd, Ts)
    cirArray = np.zeros(tsamples.size, dtype=np.complex64)

    def tRespCIR(cirList, t, Ts):
        r = np.complex64(0.0)
        # yes, this could be optimized using some sort of map
        for cirEntry in cirList:
            toa, cir = cirEntry
            dt = t - toa
            r = r + cir*np.sinc(dt/Ts)
        return r

    for i,t in enumerate(tsamples):
        cirArray[i] = tRespCIR(cirList, t, Ts)
    return cirArray, tsamples


def cirListToArray(cirList):
    tArr = np.array(list(map(lambda x: x[0], cirList)))
    cirArr = np.array(list(map(lambda x: x[1], cirList)))
    return cirArr, tArr
