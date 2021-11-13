#!/bin/bash

clients_num=$1
start_port=4000
i=1

while(( $i<=$clients_num ))
do
	port=`expr $start_port + $i`
	python app.py -h 127.0.0.1 -p $port
	let "i++"
done
