#!/usr/bin/perl

use strict;

if (scalar(@ARGV) ne 1) {
  print STDERR "perl NiuTrans.NMT-generate-unk.pl NUM-OF-UNKS\n";
  exit(1);
}


my $num_of_unks_per_line = 20;

my $num_of_unks = @ARGV[0] / $num_of_unks_per_line;

my $line_num = 0;
my $word_num = 0;
for(0..($num_of_unks - 1)) {
  for my $i (0..($num_of_unks_per_line - 1)) {
    ++$word_num;
    if ($i ne 0) {
      print " ";
    }
    print "UNK$word_num"
  }
  print "\n";
}

