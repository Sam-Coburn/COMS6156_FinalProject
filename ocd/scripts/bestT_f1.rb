#!/usr/bin/env ruby
# -*- coding: utf-8 -*-
require 'csv'
#require 'gruff'
# Author: Juan, Albert Cabré, 2014
# Edited: Chaiyong Ragkhitwetsagul, 2015

def run!
	check_args
	# read the CSV result file
	data = CSV.read(ARGV[0])
	$range = 0.05 # default is 5%
	$amount_to_remove = 100 # default is 100
	$real_TP = 500
	$real_TN = 2000
	$total_item = $real_TP + $real_TN
	$best_results = {t: -1, tp: 0, fp: 0, tn: 0, fn: 0, precision: 0.0, recall: 0.0, acc: 0.0, f1: 0.0}
	$best_t = -1	
	#$minimum_false = $total_item
	$minimum_false = 0 
	$unsure_t = 0 

	data.shift
	data = data.transpose
	index = data.shift
	
	classes = {}
	index.each_with_index do |entry,i|
		classes[i] = entry[/.*(?=\/)/]
		#classes[i] = entry
		#puts classes[i] 
	end

	max = 100 #maximum similarity = 100
	results = [] # normal results
	area = 0

	# for printing out best values	
	#puts "tool,t,min_false,fprfnr_s,unsure_t,fr,fpr,fnr,precision,recall\r"

	(0..max).each do |t|
	# iterate through each threshold from 0 to 100
		results[t] = {trueP: 0, falseP: 0, trueN: 0, falseN: 0, t: t, precision: 0, recall: 0, acc: 0.0, f1: 0}
		data.each_with_index do |row,i|
			row.each_with_index do |p,j|
				# calculate TP, TF, TN, FN
				if(classes[i] == classes[j]) # same file
					if (p.to_f <= t) # same file but treated as different (false negative)
						results[t][:falseN] += 1
					else # same file and correctly identified (true positive)
						results[t][:trueP] += 1
					end
				elsif(classes[i] != classes[j]) # different files
					if (p.to_f > t) # different file but treated as the same file (false positive)
						results[t][:falseP] += 1
					else # different files treated as different (true negative) 
						results[t][:trueN] += 1
					end	
				else
					puts "Error"
				end
			end
		end
	
		# *********************************************
		# calculate false rate, precision, recall, etc.
		# *********************************************

		if ((results[t][:trueP] + results[t][:falseP]) == 0)
			results[t][:precision] = 0
		else 
			results[t][:precision] = results[t][:trueP].to_f/(results[t][:trueP]+results[t][:falseP]).to_f
		end

		# recall
		if ((results[t][:trueP] + results[t][:falseN]) == 0)
			results[t][:recall] = 0
		else
			results[t][:recall] = results[t][:trueP].to_f/(results[t][:trueP]+results[t][:falseN]).to_f
		end
		
		# accuracy
		results[t][:acc] = (results[t][:trueP]+results[t][:trueN]).to_f/(results[t][:trueP]+results[t][:falseP]+results[t][:trueN]+results[t][:falseN]).to_f

		# f-measure
		if (results[t][:precision] +results[t][:recall] != 0)
			results[t][:f1] = 2*(results[t][:precision]*results[t][:recall])/(results[t][:precision]+results[t][:recall])
		else
			results[t][:f1] = 0;
		end
		
		# ********************************
		# Find the best result 
		# ********************************
		if (results[t][:f1] > $minimum_false)
			$minimum_false = results[t][:f1]
			$best_results[:t] = t
			$best_results[:tp] = results[t][:trueP]
			$best_results[:fp] = results[t][:falseP]
			$best_results[:tn] = results[t][:trueN]
			$best_results[:fn] = results[t][:falseN]
			$best_results[:precision] = results[t][:precision]
			$best_results[:recall] = results[t][:recall]
			$best_results[:acc] = results[t][:acc]
			$best_results[:f1] = results[t][:f1]
		end
	end

	best_threshold(results, area)
	write_results(results, ARGV[1]) # write results to CSV file
end

def best_threshold results, area
	puts "#{$best_results[:t]},#{$best_results[:tp]},#{$best_results[:fp]},#{$best_results[:tn]},#{$best_results[:fn]},#{$best_results[:precision]},#{$best_results[:recall]},#{$best_results[:acc]},#{$best_results[:f1]}"
end

def write_results results, path
	require 'fileutils'
	#check if the output dir exists. if not, create it.
	final_path = "best_t" + "/" + path
	dirname = File.dirname(final_path)
	unless File.directory?(dirname)
  		FileUtils.mkdir_p(dirname)
	end
	CSV.open(final_path +'_' + $range.to_s + '_' + $amount_to_remove.to_s + '.csv','wb') do |csv|
		csv << ['Threshold','TP', 'FP','TN', 'FN','Precision', 'Recall', 'Acc', 'F1']
		results.each do |row|
			csv << [row[:t], row[:trueP], row[:falseP], row[:trueN], row[:falseN], row[:precision], row[:recall], row[:acc], row[:f1]]
		end
	end

end

def check_args
	# check number of arguments or show usage if arguments are missing.
	if ARGV.length == 0
		puts "A Ruby script to calculate several error measures.\r\n"
		puts "==========================================================\r\n"
		puts "Usage: .//bestT_f1.rb <infile>.csv <outfile>\r\n"
		puts "<infile>.csv \t\tthe path to the matrix CSV file produced by comparing the tool's results\r"
		puts "<outfile> \t\tname of the output CSV file. Please omit .csv extension. It is added automatically.\r"
		puts "==========================================================\r\n"
		puts "Example: ./bestThreshold.rb ../diff.csv diff"
		puts "==========================================================\r\n"
		exit
	end
end

run!
