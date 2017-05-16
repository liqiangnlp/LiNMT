#!/usr/bin/perl

#######################################
#   version   : 1.0.0 Beta
#   Function  : detoken-postprocessing
#   Author    : Qiang Li
#   Email     : liqiangneu@gmail.com
#   Date      : 01/16/2017
#######################################

use strict;

my $logo =   "########### SCRIPT ########### SCRIPT ############ SCRIPT ##########\n".
             "#                                                                  #\n".
             "#  NiuTrans detoken postprocessing        liqiangneu\@gmail.com     #\n".
             "#                                                                  #\n".
             "########### SCRIPT ########### SCRIPT ############ SCRIPT ##########\n";

print STDERR $logo;

if (scalar(@ARGV) != 4) {
  print STDERR "Usage: perl Niutrans.NMT-detoken-postprocessing.pl SRC 1BEST OUTPUT LOG\n";
  exit(1);
}

open (INPUTSOURCE, "<", $ARGV[0]) or die "Error: can not read $ARGV[0]!\n";
open (INPUTTRANS, "<", $ARGV[1]) or die "Error: can not write $ARGV[1]!\n";
open (OUTPUT, ">", $ARGV[2]) or die "Error: can not write $ARGV[2]!\n";
open (OUTPUTLABEL, ">", $ARGV[2].".label") or die "Error: can not write $ARGV[2].label!\n";
open (OUTPUTLOG, ">", $ARGV[3]) or die "Error: can not write $ARGV[3]!\n";

my $line_num = 0;
my $same_line_num = 0;
my $difference_line_num = 0;
while (my $line_src = <INPUTSOURCE>) {
  ++$line_num;

  $line_src =~ s/[\r\n]//g;
  $line_src =~ s/^\s+//g;
  $line_src =~ s/\s+$//g;

  my $line_src_bak = $line_src;
  my @fields_tmp = split / \|\|\|\| /, $line_src_bak;
  $line_src = $fields_tmp[0];
  
  my $line_trans = <INPUTTRANS>;
  $line_trans =~ s/[\r\n]//g;
  $line_trans =~ s/^\s+//g;
  $line_trans =~ s/\s+$//g;
  my $line_trans_bak = $line_trans;
  @fields_tmp = split / \|\|\|\| /, $line_trans_bak;
  $line_trans = $fields_tmp[0];
  
  
  my @src_words = split /\s+/, $line_src;
  my @trans_words = split /\s+/, $line_trans;
  
  if (scalar(@src_words) ne scalar(@trans_words)) {
    ++$difference_line_num;
    print OUTPUTLOG [$line_num]."\t"."$line_src"."\t".$line_trans."\n";
    print OUTPUT "\n";
    
  } else {
    ++$same_line_num;
    my $words_number = scalar(@src_words);
    my $final_string = "";
        my $final_label = "";
    for my $i (0..($words_number - 1)) {
      if ($trans_words[$i] eq "E" or $trans_words[$i] eq "S") {
        $final_string .= $src_words[$i]." ";
        $final_label .= $trans_words[$i]." ";
      } else {
        $final_string .= $src_words[$i];
        $final_label .= $trans_words[$i];
      }
    }
    
    $final_string =~ s/\s+$//g;
    $final_string =~ s/^\s+//g;
    $final_label =~ s/\s+$//g;
    $final_label =~ s/^\s+//g;
    print OUTPUT $final_string."\n";
	print OUTPUTLABEL $final_label."\n";
    
  }
  
  if ($line_num % 10000 == 0) {
    print STDERR "\rLINE_NUM=$line_num SAME=$same_line_num DIFFERENCE=$difference_line_num";
  }
}
print STDERR "\rLINE_NUM=$line_num SAME=$same_line_num DIFFERENCE=$difference_line_num\n";
print OUTPUTLOG "\rLINE_NUM=$line_num SAME=$same_line_num DIFFERENCE=$difference_line_num\n";


close (INPUTSOURCE);
close (INPUTTRANS);
close (OUTPUT);
close (OUTPUTLABEL);
close (OUTPUTLOG);





