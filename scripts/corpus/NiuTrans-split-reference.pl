#!/usr/bin/perl -w


# * NiuTrans.NMT v1.0                   * #
# * Author: Qiang Li,                   * #
# * Email : liqiangneu@gmail.com        * #
# * Date  : 11/04/2016                  * #
# * Time  : 10:43                       * #
# * training recurrent neural network   * #


use strict;
use Encode;
use utf8;

my $logo =   "Split reference\n";

print STDERR $logo;

my %param;

GetParameter(@ARGV);

open (INPUTFILE, "<", $param{"-input"}) or die "Error: can not read $param{'-input'}\n";
open (OUTPUTFILE1, ">", $param{"-input"}.".ref1") or die "Error: can not write $param{'-input'}.ref1\n";
open (OUTPUTFILE2, ">", $param{"-input"}.".ref2") or die "Error: can not write $param{'-input'}.ref2\n";
open (OUTPUTFILE3, ">", $param{"-input"}.".ref3") or die "Error: can not write $param{'-input'}.ref3\n";
open (OUTPUTFILE4, ">", $param{"-input"}.".ref4") or die "Error: can not write $param{'-input'}.ref4\n";

my $line_num = 0;
while (my $line = <INPUTFILE>) 
{
  ++$line_num;
  $line =~ s/[\r\n]//g;
  $line =~ s/\s+$//g;
  $line =~ s/^\s+//g;
  
  if ($line_num % 4 == 1)
  {
    print OUTPUTFILE1 $line."\n";
  }
  elsif ($line_num % 4 == 2)
  {
    print OUTPUTFILE2 $line."\n";
  }
  elsif ($line_num % 4 == 3)
  {
    print OUTPUTFILE3 $line."\n";
  }
  elsif ($line_num % 4 == 0)
  {
    print OUTPUTFILE4 $line."\n";
  }
}



close (OUTPUTFILE1);
close (OUTPUTFILE2);
close (OUTPUTFILE3);
close (OUTPUTFILE4);
close (INPUTFILE);


sub GetParameter
{
  if( ( scalar( @_ ) < 2 ) || ( scalar( @_ ) % 2 != 0 ) )
  {
    print STDERR "[USAGE]\n".
                 "         NiuTrans-split-reference.pl               [OPTIONS]\n".
                 "[OPTION]\n".
                 "          -input    :  input total references file\n".
                 "[EXAMPLE]\n".
                 "         perl NiuTrans-split-reference.pl     [-input  FILE]\n";
    exit(0);
  }
          
  my $pos;
  for( $pos = 0; $pos < scalar( @_ ); ++$pos )
  {
    my $key = $ARGV[ $pos ];
    ++$pos;
    my $value = $ARGV[ $pos ];
    $param{ $key } = $value;
  }
          
  if( !exists $param{ "-input" } )
  {
    print STDERR "Error: please assign \"-input\"!\n";
    exit( 1 );
  }
}
