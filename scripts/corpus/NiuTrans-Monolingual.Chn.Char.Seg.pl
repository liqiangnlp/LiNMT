##################################################################################
#
# NiuTrans - SMT platform
# Copyright (C) 2011, NEU-NLPLab (http://www.nlplab.com/). All rights reserved.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public
# License along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
#
##################################################################################

#######################################
#   version      : 1.1.0 Beta
#   Function     : Monolingual Chinese Character Segmentation
#   Author       : Qiang Li
#   Email        : liqiangneu@gmail.com
#   Date         : 11/06/2012
#   Last Modified: 
#######################################


use strict;
use Encode;
#use utf8;

my $logo =   "########### SCRIPT ########### SCRIPT ############ SCRIPT ##########\n".
             "#                                                                  #\n".
             "# NiuTrans Mono Chn Char Seg(version 1.1.0 Beta)  --www.nlplab.com #\n".
             "#                                                                  #\n".
             "########### SCRIPT ########### SCRIPT ############ SCRIPT ##########\n";

print STDERR $logo;

my %param;

getParameter( @ARGV );

open( INFILE,  "<", $param{ "-in"  } ) or die "Error: Can not open file $param{ \"-in\"  }.\n";
open( OUTFILE, ">", $param{ "-out" } ) or die "Error: Can not open file $param{ \"-out\" }.\n";

if( $param{ "-outEncode" } eq "UTF-8" )
{
          binmode( OUTFILE, ':encoding(utf8)' );
}

my $lineNo = 0;
while( <INFILE> )
{
          s/[\r\n]//g;
          ++$lineNo;
          
          my $sentence = $_;
          if( $param{ "-inEncode" } eq "gb2312" )
          {
                    $sentence = decode( "gb2312", $_ );
          }
          elsif( $param{ "-inEncode" } eq "UTF-8" )
          {
                    $sentence = decode( "UTF-8", $_ );
          }
          
          my @chineseChar = split / +/,$sentence;
          my $word;
          my $sentenceAfterSeg;

          foreach $word ( @chineseChar )
          {
                    if( $word =~ /^[[:ascii:]]+$/ )
                    {
                              $sentenceAfterSeg = $sentenceAfterSeg.$word." ";
                              next;
                    }
                    
                    my $wordUTF8 = $word;
                    my @characters = split //,$wordUTF8;
                    my $character;
                    foreach $character ( @characters )
                    {
                              $sentenceAfterSeg = $sentenceAfterSeg.$character." ";
                    }
          }
          $sentenceAfterSeg =~ s/ +$//g;
          $sentenceAfterSeg =~ s/^ +//g;
          if( $param{ "-outEncode" } eq "gb2312" )
          {
                    print OUTFILE encode ( "gb2312", $sentenceAfterSeg ), "\n";
          }
          elsif( $param{ "-outEncode" } eq "UTF-8" )
          {
                    print OUTFILE $sentenceAfterSeg, "\n";
          }

          print STDERR "\r    Processed $lineNo lines." if( $lineNo % 10000 == 0 );
}
print STDERR "\r    Processed $lineNo lines.\n";

close( INFILE );
close( OUTFILE );

sub getParameter
{
          if( ( scalar( @_ ) < 4 ) || ( scalar( @_ ) % 2 != 0 ) )
          {
                    print STDERR "[USAGE]    :\n".
                                 "        NiuTrans-Monolingual.Chn.Char.Seg.pl        [OPTIONS]\n".
                                 "[OPTION]   :\n".
                                 "          -in          :  Inputted  File.\n".
                                 "          -out         :  Outputted File.\n".
                                 "          -inEncode    :  Inputted  File's Encode. [optional]\n".
                                 "                            Default value is 'gb2312'.\n".
                                 "                            Value is 'UTF-8', 'gb2312'.\n".
                                 "          -outEncode   :  Outputted File's Encode. [optional]\n".
                                 "                            Default value is 'gb2312'.\n".
                                 "                            Value is 'UTF-8', 'gb2312'.\n".
                                 "[EXAMPLE]  :\n".
                                 "        perl NiuTrans-Monolingual.Chn.Char.Seg.pl\n".
                                 "                                                  [-in  FILE]\n".
                                 "                                                  [-out FILE]\n";
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
          if( !exists $param{ "-in" } )
          {
                    print STDERR "Error: please assign \"-in\"!\n";
                    exit( 1 );
          }
          if( !exists $param{ "-out" } )
          {
                    print STDERR "Error: please assign \"-out\"!\n";
                    exit( 1 );
          }
          if( !exists $param{ "-inEncode" } )
          {
                    $param{ "-inEncode" } = "gb2312";
          }
          elsif( $param{ "-inEncode" } ne "gb2312" and $param{ "-inEncode" } ne "UTF-8" )
          {
                    print STDERR "Warning: Parameter '-inEncode' assigned wrong, value is 'UTF-8' or 'gb2312'.\n";
                    exit( 1 );
          }
          if( !exists $param{ "-outEncode" } )
          {
                    $param{ "-outEncode" } = "gb2312";
          }
          elsif( $param{ "-outEncode" } ne "gb2312" and $param{ "-outEncode" } ne "UTF-8" )
          {
                    print STDERR "Warning: Parameter '-outEncode' assigned wrong, value is 'UTF-8' or 'gb2312'.\n";
                    exit( 1 );
          }
}
