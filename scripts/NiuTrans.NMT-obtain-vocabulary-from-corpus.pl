#!/usr/bin/perl

#######################################
#   version   : 1.0.0 Beta
#   Function  : vocabulary
#   Author    : Qiang Li
#   Email     : liqiangneu@gmail.com
#   Date      : 02/21/2017
#######################################


use strict;

my $logo =   "########### SCRIPT ########### SCRIPT ############ SCRIPT ##########\n".
             "#                                                                  #\n".
             "#  NiuTrans Obtain Vocabulary             liqiangneu\@gmail.com     #\n".
             "#                                                                  #\n".
             "########### SCRIPT ########### SCRIPT ############ SCRIPT ##########\n";

print STDERR $logo;

if (scalar(@ARGV) != 2) {
  print STDERR "Usage: perl Niutrans.NMT-obtain-vocabulary-from-corpus.pl INPUT OUTPUT\n";
  exit(1);
}

print STDERR "Start loading $ARGV[0] ...\n";
open (INPUTFILE, "<", $ARGV[0]) or die "Error: can not read $ARGV[0]!\n";
my $line_num = 0;
my %words_dict;
my $total_words = 0;
while (my $line = <INPUTFILE>)
{
  ++$line_num;

  $line =~ s/[\r\n]//g;
  $line =~ s/^\s+//g;
  $line =~ s/\s+$//g;

  my @words = split /\s/, $line;
  foreach my $word (@words) 
  {
    ++$words_dict{$word};
    ++$total_words;
  }


  if ($line_num % 1000 == 0)
  {
    print STDERR "\r    $line_num";
  }
}
print STDERR "\r    $line_num\n";
close (INPUTFILE);
my $dict_size = keys %words_dict;
print STDERR "    DICT_SIZE=$dict_size\n";
print STDERR "    TOTAL_WORDS=$total_words\n";


print STDERR "Start output vocabulary ...\n";
open (OUTPUTFILE, ">", $ARGV[1]) or die "Error: can not write $ARGV[1]!\n";
$line_num = 0;
my $frequent = 0;
foreach my $key (sort {$words_dict{$b} <=> $words_dict{$a}} keys %words_dict) 
{
  ++$line_num;
  $frequent += $words_dict{$key};
  my $freq = $frequent / $total_words;
  $freq = sprintf("%.4f", $freq);

  print OUTPUTFILE $line_num."\t".$key."\t".$words_dict{$key}."\t$freq\n";
  
  if ($line_num % 1000 == 0)
  {
    print STDERR "\r    $line_num";
  }
}
print STDERR "\r    $line_num\n";
close (OUTPUTFILE);






