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
             "#  NiuTrans Zero-Shot                     liqiangneu\@gmail.com     #\n".
             "#                                                                  #\n".
             "########### SCRIPT ########### SCRIPT ############ SCRIPT ##########\n";

print STDERR $logo;

if (scalar(@ARGV) != 3) {
  print STDERR "Usage: perl Niutrans.NMT-zero-shot.pl LABEL INPUT OUTPUT\n";
  exit(1);
}

open (LABEL, "<", $ARGV[0]) or die "Error: can not read $ARGV[0]!\n";
my $label; 
while (my $line = <LABEL>) {
  $line =~ s/[\r\n]//g;
  $line =~ s/^\s+//g;
  $line =~ s/\s+$//g;
  $label = $line; 
}

close(LABLE);
print STDERR "LABEL=$label\n";


open (INPUT, "<", $ARGV[1]) or die "Error: can not read $ARGV[1]!\n";
open (OUTPUT, ">", $ARGV[2]) or die "Error: can not write $ARGV[2]!\n";

my $line_num = 0;
while (my $line = <INPUT>) {
  ++$line_num;
  $line =~ s/[\r\n]//g;
  $line =~ s/^\s+//g;
  $line =~ s/\s+$//g;

  my @fields = split / \|\|\|\| /, $line;
  if (scalar(@fields) eq 1) {  
    print OUTPUT $label." ".$line."\n";
  } else {
    my $generalization = "";
    if ($fields[1] =~ /^{(.*)}$/) {
      my $raw_gene = $1;
      my @generalizations = split /}{/, $raw_gene;
      foreach my $gene (@generalizations) {
        my @domains = split / \|\|\| /, $gene;
        my $new_start = $domains[0] + 1;
        my $new_end = $domains[1] + 1;
        $generalization .= "{".$new_start." ||| ".$new_end." ||| ".$domains[2]." ||| ".
                           $domains[3]." ||| ".$domains[4]."}";
      }
    } else {
      print STDERR "FORMAT ERROR in $line_num!\n";
    }

    if ($generalization ne "") {
      print OUTPUT $label." ".$fields[0]." |||| ".$generalization."\n";
    } else {
      print OUTPUT $label." ".$fields[0]."\n";
    } 



  }  

  if ($line_num % 10000 == 0) {
    print STDERR "\rLINE_NUM=$line_num";
  }
}
print STDERR "\rLINE_NUM=$line_num\n";


close (INPUT);
close (OUTPUT);





