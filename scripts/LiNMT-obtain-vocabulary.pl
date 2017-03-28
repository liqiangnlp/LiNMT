#!/usr/bin/perl

#######################################
#   version   : 1.0.0 Beta
#   Function  : zero-shot
#   Author    : Qiang Li
#   Email     : liqiangneu@gmail.com
#   Date      : 06/16/2011
#######################################


use strict;

my $logo =   "########### SCRIPT ########### SCRIPT ############ SCRIPT ##########\n".
             "#                                                                  #\n".
             "#  NiuTrans Obtain Vocabulary             liqiangneu\@gmail.com     #\n".
             "#                                                                  #\n".
             "########### SCRIPT ########### SCRIPT ############ SCRIPT ##########\n";

print STDERR $logo;

if (scalar(@ARGV) != 1) {
  print STDERR "Usage: perl Niutrans.NMT-obtain-vocabulary.pl INPUT\n";
  exit(1);
}



open (INPUT, "<", $ARGV[0]) or die "Error: can not read $ARGV[0]!\n";
open (OUTSRCVOCAB, ">", $ARGV[0].".src-vocab") or die "Error: can not write $ARGV[0].src-vocab!\n";
open (OUTTGTVOCAB, ">", $ARGV[0].".tgt-vocab") or die "Error: can not write $ARGV[0].tgt-vocab!\n";

my $line_num = 0;

<INPUT>;
<INPUT>;

my $src_flag = 1;
while (my $line = <INPUT>) {
  ++$line_num;
  $line =~ s/[\r\n]//g;
  $line =~ s/^\s+//g;
  $line =~ s/\s+$//g;
  
  if ($src_flag eq 0 && $line =~ /^=====.*=====$/) {
    last;
  } elsif ($line =~ /^=====.*=====$/) {
    $src_flag = 0;
    next;
  }

  my @fields = split /\s/, $line;
  if ($src_flag eq 1) 
  {
    print OUTSRCVOCAB $fields[1]."\n";
  }
  else
  {
    print OUTTGTVOCAB $fields[1]."\n";
  }



 
  if ($line_num % 10000 == 0) {
    print STDERR "\rLINE_NUM=$line_num";
  }
}
print STDERR "\rLINE_NUM=$line_num\n";


close (INPUT);
close (OUTSRCVOCAB);
close (OUTTGTVOCAB);





