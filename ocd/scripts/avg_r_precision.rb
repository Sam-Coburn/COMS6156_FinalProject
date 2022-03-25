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
	if (ARGV.length >= 3) # the n is given
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

	# sort the result of each row by similarity descending
	(0..0).each do |t|
	# iterate through each threshold from 0 to 100
		results[t] = {trueP: 0, falseP: 0, trueN: 0, falseN: 0, precision: 0, recall: 0, acc:0, f1: 0}
		results_sorted = []
		found_cut_off = 0

		data.each_with_index do |row,i|
			results_sorted_query = []
			row.each_with_index do |p,j|
				# collect data for sorting
				#puts i.to_s + ", " + j.to_s
				results_sorted_row = [] 
				results_sorted_row[0] = classes[i]
				results_sorted_row[1] = classes[j]
				distance = p.to_f  
				results_sorted_row[2] = p
				results_sorted_row[3] = t
				results_sorted_row[4] = distance
				#puts results_sorted_row
				
				if (results_sorted_query.count == 0)
					results_sorted_query.push(results_sorted_row) # first one, just push it
				else	
					# do insertion sort here
					inserted = false
					results_sorted_query.each_with_index do |val, id|
					if (distance >= val[4].to_f) # insert when it's the place 
						results_sorted_query.insert(id, results_sorted_row)
						inserted = true
						break
					end
				end
				if (!inserted) # put it at the end
					results_sorted_query.push(results_sorted_row)
				end
			end
		end
		#puts results_sorted_query.to_s
		results_sorted.push(results_sorted_query)
	end

	# >>>>>>>>>>>>>>>>>  TO ADD REMOVAL REMOVE THIS COMMENT ==========================
		
	#
	count = 1
	$false_sort_results = [] 
	sum_r_precision = 0

	# puts "Size of results_sorted = " + results_sorted.size.to_s

	while count <= results_sorted.size do # while still less than the specified number
		fpCount = 0
		fnCount =0
		tpCount = 0
		tnCount = 0
		simval = -1
		val = results_sorted[count-1] # get result of each row (a single query) out of the sorted array

		element_count = 0
		# get only the top r results 
		while element_count < $MAX_RESULTS.to_i do
			x = val[element_count]
			# calculate TP, TF, TN, FN
			simval = x[2].to_f
			if(x[0].split('/')[0] == x[1].split('/')[0]) # same file
	        	tpCount += 1
	 		elsif(x[0] != x[1]) # different files
	 			fpCount += 1
			else
	   			puts "Error"
			end
			element_count += 1
		end

		rPrec = tpCount.to_f/$MAX_RESULTS.to_i
		false_sort_row = []
		false_sort_row[0] = count.to_s
		false_sort_row[1] = tpCount.to_s
		false_sort_row[2] = fpCount.to_s
		false_sort_row[3] = rPrec.to_s

		sum_r_precision += rPrec
		#puts "r-prec," + count.to_s + "," + tpCount.to_s + "," + fpCount.to_s + "," + rPrec.to_s
		$false_sort_results.push(false_sort_row)
		count=count+1
	end

	avg_sum_r_precision = sum_r_precision.to_f/results_sorted.size
	#$FINAL_VAL = tpCount.to_s + "," + fpCount.to_s + "," + avg_sum_r_precision.round(4).to_s
	puts avg_sum_r_precision.round(4).to_s + "\n"
	write_results($false_sort_results, ARGV[1])
end
end

def write_results results, path
	#puts "Writing results ... "
	# Write results to a file
	require 'fileutils'
	#check if the output dir exists. if not, create it.
	final_path = "arp/" + ARGV[1] 
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
		puts "A Ruby script to calculate Average r-Precision (ARP).\r\n"
		puts "==========================================================\r\n"
		puts "Usage: ./avg_r_precision.rb <infile>.csv <outfile> <amount>\r\n"
		puts "<infile>.csv \t\tthe path to the matrix CSV file produced by comparing the tool's results\r"
		puts "<outfile> \t\tname of the output CSV file. Please omit .csv extension. It is added automatically.\r"
		puts "<amount> \t\tmaximum amount (r) to check.\r\n"
		puts "==========================================================\r\n"
		puts "Example: ./avg_r_precision.rb ../diff.csv diff 10"
		puts "==========================================================\r\n"
		exit
	end
end

run!
