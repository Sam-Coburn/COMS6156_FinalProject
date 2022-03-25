#!/usr/bin/env ruby
# -*- coding: utf-8 -*-
require 'csv'
#require 'gruff'
# Author: Chaiyong Ragkhitwetsagul, 2017

def run!
	check_args
	# read the CSV result file
	data = CSV.read(ARGV[0])
	
	$real_TP = 1000
	$real_TN = 9000
	$TOTAL = 10000

	$total_item = $real_TP + $real_TN
	$best_results = {t: -1, fp: 0, fn: 0, precision: 0.0, recall: 0.0, acc: 0.0}
	$best_t = -1	
	$minimum_false = $total_item
	$unsure_t = 0 

	$MAX_RESULTS = 10000 # intial value
	if (ARGV.length >= 3) # the number of n is given
		$MAX_RESULTS = ARGV[2]
	end

	data.shift
	data = data.transpose
	index = data.shift

	$file_amount = 0
	
	classes = {}
	index.each_with_index do |entry,i|
		classes[i] = entry
		$file_amount = $file_amount+1
	end
	
	$total_comparison = $file_amount * $file_amount

	# for printing out best values	
	max = 100
	results = []

	# sort the result by similarity descending
	(0..0).each do |t|
		# iterate through each threshold from 0 to 100
		results[t] = {trueP: 0, falseP: 0, trueN: 0, falseN: 0, precision: 0, recall: 0, acc:0, f1: 0}
		results_sorted = []
		found_cut_off = 0

		data.each_with_index do |row,i|
			row.each_with_index do |p,j|
				# collect data for sorting
				results_sorted_row = [] 
				results_sorted_row[0] = classes[i]
				results_sorted_row[1] = classes[j]
				distance = p.to_f  
				results_sorted_row[2] = p
				results_sorted_row[3] = t
				results_sorted_row[4] = distance
				#puts results_sorted_row
				
				if (results_sorted.count == 0)
					results_sorted.push(results_sorted_row) # first one, just push it
				else	
					# do insertion sort here
					inserted = false
					results_sorted.each_with_index do |val, id|
					if (distance >= val[4].to_f) # insert when it's the place 
						results_sorted.insert(id, results_sorted_row)
						inserted = true
						break
					end
				end
				if (!inserted) # put it at the end
					results_sorted.push(results_sorted_row)
				end
			end
		end
	end

	# >>>>>>>>>>>>>>>>>  TO ADD REMOVAL REMOVE THIS COMMENT ==========================
		
	#
	count = 1
	$false_sort_results = [] 
	#puts count
	fpCount = 0
	fnCount =0
	tpCount = 0
	tnCount = 0
	simval = -1

	while count <= $MAX_RESULTS.to_i do # while still less than the specified number
		val = results_sorted[count-1] # get the value out of the sorted array

		# calculate TP, TF, TN, FN
		simval = val[2].to_f
		if(val[0].split('/')[0] == val[1].split('/')[0]) # same file
        	tpCount += 1
 		elsif(val[0] != val[1]) # different files
 			fpCount += 1
		else
   			puts "Error"
		end

		tn = ($TOTAL.to_i - $MAX_RESULTS.to_i) - fpCount
		fn = $MAX_RESULTS.to_i - tpCount
		precAtN = tpCount.to_f/$MAX_RESULTS.to_i

		false_sort_row = []
		false_sort_row[0] = count.to_s
		false_sort_row[1] = tpCount.to_s
		false_sort_row[2] = fpCount.to_s

		false_sort_row[3] = precAtN.to_s

		$false_sort_results.push(false_sort_row)
		$FINAL_VAL = tpCount.to_s + "," + fpCount.to_s + "," + precAtN.round(4).to_s 

		count=count+1
	end

	puts $FINAL_VAL + "\n"
	write_results($false_sort_results, ARGV[1])
end
end

def write_results results, path
	#puts "Writing results ... "
	# Write results to a file
	require 'fileutils'
	#check if the output dir exists. if not, create it.
	final_path = "prec-at-n/" + ARGV[1] 
	#puts final_path
	dirname = File.dirname(final_path)
	unless File.directory?(dirname)
		FileUtils.mkdir_p(dirname)
	end

	CSV.open(final_path + '.csv','wb') do |csv|
		csv << ['count','tp','fp','tn','fn','prec']
		results.each do |row|
			csv << [row[0],row[1],row[2],row[3],row[4],row[5]]
		end
	end
end

def check_args
	# check number of arguments or show usage if arguments are missing.
	if ARGV.length == 0
		puts "A Ruby script to calculate Precision-at-n.\r\n"
		puts "==========================================================\r\n"
		puts "Usage: ./prec-at-n.rb <infile>.csv <outfile> <amount>\r\n"
		puts "<infile>.csv \t\tthe path to the matrix CSV file produced by comparing the tool's results\r"
		puts "<outfile> \t\tname of the output CSV file. Please omit .csv extension. It is added automatically.\r"
		puts "<amount> \t\tthe n amount to check.\r\n"
		puts "==========================================================\r\n"
		puts "Example: ./prec_at_n.rb ../diff.csv diff 100"
		puts "==========================================================\r\n"
		exit
	end
end

run!
