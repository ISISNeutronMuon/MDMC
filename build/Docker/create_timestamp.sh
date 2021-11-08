#!/bin/sh

# creates a file called TIMESTAMP with container debug info.
echo "This docker image was created on: " > TIMESTAMP
echo `date` >> TIMESTAMP && \
echo "With the following requirements.txt: " >> TIMESTAMP
cat ./requirements.txt | while read line; do
	echo "$line" >> TIMESTAMP
	done

