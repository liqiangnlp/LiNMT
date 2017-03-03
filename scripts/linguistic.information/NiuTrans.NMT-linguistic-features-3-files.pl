#!/usr/bin/perl

#######################################
#   version   : 1.0.0 Beta
#   Function  : linguistic-features-3-files
#   Author    : Qiang Li
#   Email     : liqiangneu@gmail.com
#   Date      : 01/16/2017
#######################################

use strict;

my $logo =   "########### SCRIPT ########### SCRIPT ############ SCRIPT ##########\n".
             "#                                                                  #\n".
             "#  NiuTrans linguistic features 3 files   liqiangneu\@gmail.com     #\n".
             "#                                                                  #\n".
             "########### SCRIPT ########### SCRIPT ############ SCRIPT ##########\n";

print STDERR $logo;

if (scalar(@ARGV) != 4) {
  print STDERR "Usage: perl Niutrans.NMT-linguistic-features-3-files.pl POS NER CHUNK OUTPUT\n";
  exit(1);
}


open (INFEAT1, "<", $ARGV[0]) or die "Error: can not read $ARGV[0]!\n";
open (INFEAT2, "<", $ARGV[1]) or die "Error: can not read $ARGV[1]!\n";
open (INFEAT3, "<", $ARGV[2]) or die "Error: can not read $ARGV[2]!\n";
open (OUTPUT, ">",  $ARGV[3]) or die "Error: can not write $ARGV[3]!\n";
open (OUTPUTLOG, ">", $ARGV[3].".log") or die "Error: can not write $ARGV[3].log!\n";


my $line_num = 0;
my $error_num = 0;
while (my $line_feat1 = <INFEAT1>) {
  $line_feat1 =~ s/[\r\n]//g;
  $line_feat1 =~ s/^\s+//g;
  $line_feat1 =~ s/\s+$//g;

  ++$line_num;

  my $line_feat2 = <INFEAT2>;
  $line_feat2 =~ s/[\r\n]//g;
  $line_feat2 =~ s/^\s+//g;
  $line_feat2 =~ s/\s+$//g;

  my $line_feat3 = <INFEAT3>;
  $line_feat3 =~ s/[\r\n]//g;
  $line_feat3 =~ s/^\s+//g;
  $line_feat3 =~ s/\s+$//g;
  
  my @feat1_words = split /\s+/, $line_feat1;
  my @feat2_words = split /\s+/, $line_feat2;
  my @feat3_words = split /\s+/, $line_feat3;
  
  
  if (scalar(@feat1_words) ne scalar(@feat2_words) or scalar(@feat1_words) ne scalar(@feat3_words)) {
    ++$error_num;
    print STDERR "Error: the number of features is not the same in $line_num!\n";
    print OUTPUTLOG "[$line_num]\t[".scalar(@feat1_words)."] $line_feat1\t[".
                    scalar(@feat2_words)."] $line_feat2\t[".
                    scalar(@feat3_words)."] $line_feat3\n";

   print OUTPUT "N-A-POS N-A-NER N-A-CHK\n";
   next;
  }
  
  my $features_number = split /\s+/, $line_feat1;
  my $final_output = "";
  for my $i (0..$features_number - 1) {
    $final_output .= $feat1_words[$i]."-POS ".$feat2_words[$i]."-NER ".$feat3_words[$i]."-CHK ";
  }
  
  $final_output =~ s/[\r\n]//g;
  $final_output =~ s/\s+$//g;
  $final_output =~ s/^\s+//g;
  
  print OUTPUT $final_output."\n";
  
  
  if ($line_num % 10000 == 0) {
    print STDERR "\rLINE_NUM=$line_num ERROR_NUM=$error_num";
  }
}

print OUTPUTLOG "\nLINE_NUM=$line_num ERROR_NUM=$error_num\n";
print STDERR "\rLINE_NUM=$line_num ERROR_NUM=$error_num\n";


close (INFEAT1);
close (INFEAT2);
close (INFEAT3);
close (OUTPUT);
close (OUTPUTLOG);




