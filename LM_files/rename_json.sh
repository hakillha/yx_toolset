#!/bin/sh


for _f in $(ls); do
	_fname=$(jq .path ./$_f | sed 's/.*\\//g;s/.*\///g;s/"//')
	_suffix=${_fname##*.}
	_bname=$(basename ./$_fname .${_suffix})
	
	if echo $_bname | grep -q '+'; then
		mv -fv $_f ${_bname}.json
	fi
done
