#!/usr/bin/perl -w


# * NiuTrans.NMT v1.0                   * #
# * Author: Qiang Li,                   * #
# * Email : liqiangneu@gmail.com        * #
# * Date  : 10/30/2016                  * #
# * Time  : 15:31                       * #
# * training recurrent neural network   * #


use strict;
use Encode;
use utf8;

my $logo =   "Length Restrict\n";

print STDERR $logo;

my %param;


GetParameter( @ARGV );


open( SRCINFILE, "<", $param{ "-chn" } ) or die "Error: can not read file $param{ '-chn' }.\n";
open( TGTINFILE, "<", $param{ "-eng" } ) or die "Error: can not read file $param{ '-eng' }.\n";




sub GetParameter
{
  if( ( scalar( @_ ) < 4 ) || ( scalar( @_ ) % 2 != 0 ) )
  {
    print STDERR "[USAGE]\n".
                 "         NiuTrans.NMT-bilingual-max-length.pl      [OPTIONS]\n".
                 "[OPTION]\n".
                 "          -chn    :  Input chinese file.\n".
                 "          -eng    :  Input english file.\n".
                 "          -outchn :  Output filtered chinese file.\n".
                 "          -outeng :  Output filtered english file.\n".
                 "       -maxlensrc :  Max length for chn.\n".
                 "                       Default is 100 characters.\n".
                 "       -maxlentgt :  Max length for eng.\n".
                 "                       Default is 50 words"
                 "[EXAMPLE]\n".
                 "     perl NiuTrans.NMT-bilingual-max-length.pl  [-chn    FILE]\n".
                 "                                                [-eng    FILE]\n".
                 "                                                [-outchn FILE]\n".
                 "                                                [-outeng FILE]\n";
    exit( 0 );
  }
          
  my $pos;
  for( $pos = 0; $pos < scalar( @_ ); ++$pos )
  {
    my $key = $ARGV[ $pos ];
    ++$pos;
    my $value = $ARGV[ $pos ];
    $param{ $key } = $value;
  }
          
  if( !exists $param{ "-chn" } )
  {
    print STDERR "Error: please assign \"-chn\"!\n";
    exit( 1 );
  }

  if( !exists $param{ "-eng" } )
  {
    print STDERR "Error: please assign \"-eng\"!\n";
    exit( 1 );
  }

  if( !exists $param{ "-outchn" } )
  {
    print STDERR "Error: please assign \"-outchn\"!\n";
    exit( 1 );
  }

  if( !exists $param{ "-outeng" } )
  {
    print STDERR "Error: please assign \"-outeng\"!\n";
    exit( 1 );
  }

  if( !exists $param{ "-maxlensrc" } )
  {
    $param{ "-maxlensrc" } = 100;
  }

  if( !exists $param{ "-maxlentgt" } )
  {
    $param{ "-maxlentgt" } = 50;
  }  

}
