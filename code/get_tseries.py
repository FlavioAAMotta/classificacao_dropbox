#! /usr/bin/python3

import sys
import time
import pandas as pd
from collections import defaultdict
from optparse import OptionParser
from functools import reduce


#*******************************************************************************
# Constant
#*******************************************************************************
#f_changed(0) now(1) delta_change(2) delta_seen(3) user_id(4) host_int(5) client_ip(6) folder(7) vers1(8) vers2(9) ses_on(10) vol(11) type(12)
FLAG, TIME, DCHG, DSEE, USER, HOST, IP, NAMESPACE, VER1, VER2, S_ON, VOL, TYPE = range(0, 13)
WIN_LEN = 4


#*******************************************************************************
# Functions
#*******************************************************************************
def process_opt():
    usage = "usage: %prog [options] files\n"
    parser = OptionParser(usage=usage)
    parser.add_option("-i", "--infile", dest="INFILE", help="infile")    
    parser.add_option("-o", "--outfile_id", dest="OUTFILE_ID", help="outfile")
    opt, files = parser.parse_args()
    if not(opt.INFILE and opt.OUTFILE_ID):
        parser.print_help()
        sys.exit(1)
    return opt, files


#*******************************************************************************
# Main:
#*******************************************************************************
if __name__ == '__main__':
    opt, files = process_opt()

    nm_set = set()
    time_set = set()
    acc_dict = defaultdict(int)
    vol_dict = defaultdict(float)
    
    for line in open(opt.INFILE):
        data = line.split()
        if (data[FLAG] == 'f_changed'):
            t = float(data[TIME])
            nm = data[NAMESPACE]
            time_set.add( int(t) )
            nm_set.add( int(nm) )
            key = nm + '.' + time.strftime("%W_%Y", time.gmtime(t))
            acc_dict[key] += 1
            if data[VOL] != '-':
                vol_dict[key] += float(data[VOL])

    ##tests
    #t1 = time.strptime("07 Mar 21", "%d %b %y")
    #t2 = time.strptime("27 Mar 21", "%d %b %y")
    #time_set = set([int(time.mktime(t1)), int(time.mktime(t2))])

    # make the dataframe columns header (week_year)
    i = min(time_set)
    i_nth = max(time_set)
    columns_set = set( [time.strftime("%Y_%W", time.gmtime(i))] )
    while i < i_nth:
        i += 86400  # add one day 24*60*60 sec
        columns_set.add( time.strftime("%Y_%W", time.gmtime(i)) )
    columns_list = []
    for item in sorted(columns_set):    #sort the set by year,week order
        year, week = item.split('_')
        columns_list.append( week + '_' + year )
    
    df_acc = pd.DataFrame(0, index=list(nm_set), columns=columns_list )
    df_acc.index.name = 'NameSpace'
    
    df_cls = pd.DataFrame(0, index=list(nm_set), columns=columns_list )
    df_cls.index.name = 'NameSpace'    
    
    df_vol = pd.DataFrame(0.0, index=list(nm_set), columns=columns_list )
    df_vol.index.name = 'NameSpace'    
    
    for key, value in acc_dict.items():
        nm, week = key.split('.')
        df_acc.at[int(nm), week] = value
        
    for col in columns_list:
        row_idx = df_acc.loc[:, col] > 0
        df_cls.loc[ row_idx, col ] = 1
        
    for key, value in vol_dict.items():
        nm, week = key.split('.')
        df_vol.at[int(nm), week] = value
    df_vol = df_vol.cumsum(axis=1)
    
    nm_set = set()
    nm_cum_list = list()
    n_windows = (df_acc.shape[1] // WIN_LEN) + (df_acc.shape[1] % WIN_LEN > 0)
    for window in range(n_windows): # 4 columns per window
        ini = (window)*4
        fim = (window+1)*4
        df_acc_win = df_acc.iloc[:, ini:fim]
        df_acc_win = pd.Series( index=df_acc_win.index,
            data=[reduce((lambda x,y: x + y), [int(val) for val in accs]) for nm, accs in df_acc_win.iterrows()])
        nm_set.update( df_acc_win[ df_acc_win > 0 ].index )
        nm_cum_list.append( nm_set.copy() )
        
    f = open('nsJanelas_' + opt.OUTFILE_ID + '.txt', 'w')
    for item in nm_cum_list:
        f.write( ' '.join(map(str, item)) + '\n' )
    f.close()
        
    df_acc.to_csv('access_' + opt.OUTFILE_ID + '.txt', sep=" ", index_label=df_acc.index.name)
    df_cls.to_csv('target_' + opt.OUTFILE_ID + '.txt', sep=" ", index_label=df_cls.index.name)
    df_vol.to_csv('vol_bytes_' + opt.OUTFILE_ID + '.txt', sep=" ", index_label=df_vol.index.name)
    
    
