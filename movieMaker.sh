#!/usr/bin/env bash

function movieMaker {
  echo $1
  convert -delay 40 $(ls $1.* | sort -n -t. -k2) $1.gif
}

movieMaker "wealthDistribution"
movieMaker "individual"
