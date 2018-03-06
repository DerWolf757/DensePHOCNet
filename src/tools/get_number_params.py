'''
Created on Dec 4, 2017

@author: fwolf
'''
import argparse
import numpy as np
from numpy import prod, sum
from pprint import pprint

from phocnet.io.context_manager import Suppressor
import caffe


def main():
    
    parser = argparse.ArgumentParser()
    
    # required training parameters
    parser.add_argument('--net_file', '-f', action='store', type=str,
                      help='The location of the net file.')
    parser.add_argument('--dense', '-d', action='store_true',
                        help='dense network')
 
    params = vars(parser.parse_args())
    
    caffe.set_mode_cpu()
    
    deploy_file = params["net_file"]
    
    
    print "Net: " + deploy_file
    
    with Suppressor():
        net = caffe.Net(deploy_file, caffe.TEST)
        
    print "Layer-wise parameters: "
    pprint([(k, v[0].data.shape, prod(v[0].data.shape) ) for k, v in net.params.items()])
    
    total = sum([prod(v[0].data.shape) for k, v in net.params.items()])
    
    if params["dense"]:
        fc = sum([prod(net.params["fc6_d"][0].data.shape),prod(net.params["fc7_d"][0].data.shape),prod(net.params["fc8_d"][0].data.shape)])
    else:
        fc = sum([prod(net.params["fc6"][0].data.shape),prod(net.params["fc7"][0].data.shape),prod(net.params["fc8"][0].data.shape)])
    
    conv = total-fc
    
    print "Total number of parameters: " + str(total)
    print "Convolutional: " + str(conv)
    print "FC: " + str(fc)
    
    if params["dense"]:
        print "Depth at Tpp: " + str(net.params["fc6_d"][0].data.shape[1]/15)
    else:
        print "Depth at Tpp: " + str(net.params["fc6"][0].data.shape[1]/15)
        
def get_number_of_params(net):
    sum([prod(v[0].data.shape) for k, v in net.params.items()])
  
    



if __name__ == '__main__':
    main()