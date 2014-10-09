#!/usr/bin/env bash
set -e
# set -x

user_command=$1

# Show a process' properties wrt memory sizes and page faults.
# Additional options: -a
# Virtual set | resident set | minor page faults | major page faults | command |
ps_command="ps -e -o vsize,rss,minflt,majflt,cmd | grep --regexp $user_command | grep --invert-match grep | grep --invert-match `basename $0`"

eval $ps_command
while(true); do
    sleep 1
    eval $ps_command
done
