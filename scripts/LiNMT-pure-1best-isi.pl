#!/usr/bin/perl

#######################################
#   version   : 1.0.0 Beta
#   Function  : generate-1-best
#   Author    : Qiang Li
#   Email     : liqiangneu@gmail.com
#   Date      : 06/16/2011
#######################################


use strict;

my $logo =   "########### SCRIPT ########### SCRIPT ############ SCRIPT ##########\n".
             "#                                                                  #\n".
             "#  NiuTrans Pure 1-best                   liqiangneu\@gmail.com     #\n".
             "#                                                                  #\n".
             "########### SCRIPT ########### SCRIPT ############ SCRIPT ##########\n";

print STDERR $logo;

if( scalar( @ARGV ) != 2 )
{
    print STDERR "Usage: perl Niutrans-pure-1-best-isi.pl isi-results 1-best!\n";
    exit(1);
}

open( NBEST, "<".$ARGV[0] ) or die "Error: Can not open file $ARGV[0]\n";
open( ONEBEST, ">".$ARGV[1] ) or die "Error: Can not open file $ARGV[1]\n";

my $start_flag = 0;
my $line_num = 0;
while(my $line = <NBEST> )
{
  ++$line_num;
  if ($line_num % 100 == 0)
  {
    print STDERR "\rLINE_NUM=$line_num";
  }

  $line =~ s/[\r\n]//g;
  $line =~ s/^\s+//g;
  $line =~ s/\s+$//g;

  if ($start_flag eq 1) 
  {
    my $line_bak = $line;
    if ($line_bak =~ /^<START>(.*)<EOF>$/)
    {
      $line = $1;
    } 
    else
    {
      print STDERR "\nFORMAT ERROR in $line_num\n";
    } 

    $line =~ s/^\s+//g;
    $line =~ s/\s+$//g;    
    print ONEBEST "$line\n";
    $start_flag = 0;
    next;
  }


  if($line =~ /^---*---$/) 
  {
    $start_flag = 1;
    next;
  }
  
  if ($start_flag eq 0)
  {
    next;
  }

}

print STDERR "\rLINE_NUM=$line_num\n";


close( NBEST );
close( ONEBEST );
