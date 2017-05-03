#!/usr/bin/perl

use strict;

if (scalar(@ARGV) ne 2) {
  print STDERR "NiuTrans.NMT-word-label.pl INPUT OUTPUT\n";
  exit(1);
}

open (INPUTFILE, "<", $ARGV[0]) or die "Error: can not read $ARGV[0]!\n";
open (OUTPUT, ">", $ARGV[1]) or die "Error: can not write $ARGV[1]!\n";


my $illegal_num = 0;
my $line_num = 0;
my $verb_num = 0;
my $unverb_num = 0;
while(my $line = <INPUTFILE>)
{
  ++$line_num;
  $line =~ s/[\r\n]//g;
  my @all_word_and_pos = split /\s+/, $line;
  my $i = 0;
  my $j = 0;
  foreach my $word_and_pos (@all_word_and_pos) {
    if($word_and_pos =~ /^(.*)\/(.*)$/) {

      my $label = $2;
      my $word = $1;
      
      if($i ne 0) {
        print OUTPUT " ";
      }
      if ($label =~ /^VB/ or $label eq "," or $label eq ".")
      {
        ++$verb_num;
        print OUTPUT $word;
      } 
      else 
      {
        ++$j;
        ++$unverb_num;
        print OUTPUT "\#$j";
      }
      ++$i;
    } else {
      ++$illegal_num;
    }
  }
  print OUTPUT "\n";
  if ($line_num % 10000 == 0)
  {
    my $percentage = $verb_num / $unverb_num;
    print STDERR "\r$line_num sentences, $illegal_num is not right, p = $percentage";
  }
}
my $percentage = $verb_num / $unverb_num;
print STDERR "\r$line_num sentences, $illegal_num is not right, p = $percentage\n";



close (INPUTFILE);
close (OUTPUT);
