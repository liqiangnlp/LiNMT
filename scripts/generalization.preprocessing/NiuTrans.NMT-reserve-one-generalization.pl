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
             "#  NiuTrans Reserve One Generalization       liqiangneu\@gmail.com  #\n".
             "#                                                                  #\n".
             "########### SCRIPT ########### SCRIPT ############ SCRIPT ##########\n";

print STDERR $logo;

if (scalar(@ARGV) != 2) {
  print STDERR "Usage: perl Niutrans.NMT-reserve-one-generalization.pl INPUT OUTPUT\n";
  exit(1);
}



open (INPUT, "<", $ARGV[0]) or die "Error: can not read $ARGV[1]!\n";
open (OUTPUT, ">", $ARGV[1]) or die "Error: can not write $ARGV[2]!\n";

my $line_num = 0;
while (my $line = <INPUT>) {
  ++$line_num;
  $line =~ s/[\r\n]//g;
  $line =~ s/^\s+//g;
  $line =~ s/\s+$//g;

  if ($line eq "") {
    print OUTPUT "\n";
    next;
  }

  my @fields = split / \|\|\|\| /, $line;
  if (scalar(@fields) eq 1) {
    print OUTPUT $line."\n";
  } else {

    my @src_words = split /\s/, $fields[0];
    

    my $generalization = "";
    if ($fields[1] =~ /^{(.*)}$/)
    {
      my $raw_gene = $1;
      my @generalizations = split /}{/, $raw_gene;
      my %position;
      foreach my $gene (@generalizations) 
      {
        my @domains = split / \|\|\| /,$gene;
        if (exists $position{$domains[0]}) 
        {
          next;
        } 
        else 
        {
  
#          print STDERR $src_words[$domains[0]]."\t".$domains[3]."\n";
#          exit;
          if ($src_words[$domains[0]] eq $domains[3])
          {
            $generalization .= "{".lc($gene)."}";
          }
          $position{$domains[0]} = 1;
        }
      }
#      print OUTPUT $generalization."\n";
    } 
    else {
      print STDERR "FORMAT ERROR in $line_num!\n";
    }
   

    if ($generalization ne "") 
    {
      print OUTPUT $fields[0]." |||| ".$generalization."\n";
    }
    else
    {
      print OUTPUT $fields[0]."\n";
    }
  }
  
  
  if ($line_num % 10000 == 0) {
    print STDERR "\rLINE_NUM=$line_num";
  }
}
print STDERR "\rLINE_NUM=$line_num\n";


close (INPUT);
close (OUTPUT);





