#!/usr/bin/perl

use strict;

if (scalar(@ARGV) ne 3) {
  print STDERR "NiuTrans.NMT-word-label.pl INPUT WORD LABEL\n";
  exit(1);
}

open (INPUTFILE, "<", $ARGV[0]) or die "Error: can not read $ARGV[0]!\n";
open (OUTPUTWORD, ">", $ARGV[1]) or die "Error: can not write $ARGV[1]!\n";
open (OUTPUTLABEL, ">", $ARGV[2]) or die "Error: can not write $ARGV[2]!\n";


my $illegal_num = 0;
my $line_num = 0;
while(my $line = <INPUTFILE>)
{
  ++$line_num;
  $line =~ s/[\r\n]//g;
  my @all_word_and_pos = split /\s+/, $line;
  my $i = 0;
  foreach my $word_and_pos (@all_word_and_pos) {
    if($word_and_pos =~ /^(.*)\/(.*)$/) {
      if($i ne 0) {
        print OUTPUTLABEL " ";
        print OUTPUTWORD " ";
      }
      print OUTPUTLABEL $2;
      print OUTPUTWORD $1;
      ++$i;
    } else {
      ++$illegal_num;
    }
  }
  print OUTPUTLABEL "\n";
  print OUTPUTWORD "\n";
  if ($line_num % 10000 == 0)
  {
    print STDERR "\r$line_num sentences, $illegal_num is not right";
  }
}
print STDERR "\r$line_num sentences, $illegal_num is not right\n";



close (INPUTFILE);
close (OUTPUTWORD);
close (OUTPUTLABEL);
