'''
Created on Mar 11, 2013

@author: lrothack
'''

from collections import defaultdict

class WordHistogram(object):
    '''
    classdocs
    '''

        
    @staticmethod
    def linelist_to_wordlist(line_list):
        '''
        Converts the given line_list to a list of words 
        
        @param line_list: A list of lines. Each line may contain an arbitrary 
            number of words
        
        '''
        words = ' '.join(line_list)
        word_list = words.split()
        return word_list
        
    @staticmethod
    def histogram(word_list):
        '''
        Builds a histogram of words contained in the given line_list
        
        @param line_list: A list of words.
        
        '''
        
        words_hist = defaultdict(int)
        
        for w in word_list:
            words_hist[w] += 1
            
        return words_hist