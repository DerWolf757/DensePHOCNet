'''
Created on Nov 29, 2017

@author: fwolf
'''
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pyinotify
import time

class ModHandler(pyinotify.ProcessEvent):
    
    def my_init(self, file):
        self.log_file = file
        
    def process_default(self, event):
        visu_log_file(self.log_file)
       

def main():
    parser = argparse.ArgumentParser()
    
    # required training parameters
    parser.add_argument('--log_file', '-f', action='store', type=str, required=True,
                      help='The location of the log file.')
    parser.add_argument('--monitor', '-m', action='store_true', default=False,
                      help='keep monitoring the specified log file')
    parser.add_argument('--save_fig', '-s', action='store_true', default=False,
                      help='save plots')
    parser.add_argument('--fig_name', '-n', action='store', type=str, default='new_fig',
                      help='Name of created figures')
    
    params = vars(parser.parse_args())

    visu_log_file(params["log_file"], save=params["save_fig"], fig_name = params["fig_name"])
    
    if params["monitor"]:
        start_notifier(params["log_file"])
    else:
        plt.ioff()
        plt.show()
    

    
def start_notifier(log_file): 
    mask = pyinotify.IN_MODIFY
    
    handler = ModHandler(file = log_file)
    wm = pyinotify.WatchManager()
    
    notifier = pyinotify.Notifier(wm, handler)
    wdd = wm.add_watch(log_file, mask)
    
    notifier.loop()



def visu_log_file(log_file, save=False, fig_name='new_fig'):

    mAPs = np.array([])
    mAPs_iter = np.array([])
    
    loss = np.array([])
    loss_iter = np.array([])
    
    
    # read log file
    with open(log_file) as f:
            lines = f.readlines()
            
    for k,l in enumerate(lines):
        idx_map = l.find('mAP')
        if idx_map != -1:
            try:
                mAPs = np.append(mAPs, float(l[idx_map + 5:-1])*100)
                splits = lines[k+1].split()
                mAPs_iter = np.append(mAPs_iter, 0) if splits[5].find('running') != -1 else np.append(mAPs_iter, int(splits[5][:-1]))
            except ValueError:
                pass
        
        idx_loss = l.find('loss')           
        if idx_loss != -1 and l.find('Iteration') != -1:
            try:
                loss = np.append(loss, float(l[idx_loss+7:-1]))
                splits = l.split()
                loss_iter = np.append(loss_iter, int(splits[5]))
            except:
                pass
    
    
    # visu mAP and loss
    plt.ion()
    
    plt.figure(1)
    plt.plot(loss_iter,loss)
    plt.xlabel('Training Iteration')
    plt.ylabel('Loss')
    
    if save:
        plt.savefig('loss_'+fig_name+'.png')
        

    plt.figure(2)
    print np.shape(mAPs[:np.size(mAPs_iter)])
    plt.plot(mAPs_iter,mAPs[:np.size(mAPs_iter)])
    plt.xlabel('Training Iteration')
    plt.ylabel('mAP [%]')
    
    if save:
        plt.savefig('mAP_' + fig_name + '.png')
    
    plt.pause(0.1)
    
    plt.show(block=False)
 
    
         

if __name__ == '__main__':
    main()

